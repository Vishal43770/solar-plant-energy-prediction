import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns # Optional, but often imported alongside
import os # For checking file existence

# --- Configuration ---
DATA_PATH1 = '/home/vishal/office/databricks/datasets/Combined_Generation_Weather_Data1.csv'
DATA_PATH2 = '/home/vishal/office/databricks/datasets/Combined_Generation_Weather_Data2.csv'

# Columns the user will PROVIDE for the future
USER_PROVIDED_FUTURE_COLS = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
# Columns the models will PREDICT based on user input
MODEL_PREDICTED_COLS = ['DC_POWER', 'AC_POWER']
# Columns that will be DERIVED from predictions
DERIVED_COLS = ['DAILY_YIELD_forecasted', 'TOTAL_YIELD_forecasted']

N_HOURS_TO_FORECAST = 24
FORECAST_INTERVAL_MINUTES = 15
RECORDS_PER_HOUR_FORECAST = 60 // FORECAST_INTERVAL_MINUTES
NUM_FUTURE_PERIODS = N_HOURS_TO_FORECAST * RECORDS_PER_HOUR_FORECAST

# --- 1. Load and Combine Data ---
print("--- 1. Loading and Combining Historical Data ---")
try:
    df1 = pd.read_csv(DATA_PATH1)
    df2 = pd.read_csv(DATA_PATH2)
    df_historical = pd.concat([df1, df2], ignore_index=True)
except FileNotFoundError:
    print(f"Error: Historical CSV files not found at specified paths.")
    exit()
print(f"Shape of combined historical data: {df_historical.shape}")

# --- 2. Initial Exploration & Cleaning of Historical Data ---
print("\n--- 2. Initial Exploration & Cleaning of Historical Data ---")
try:
    df_historical['DATE_TIME'] = pd.to_datetime(df_historical['DATE_TIME'])
except Exception as e:
    print(f"Error parsing DATE_TIME in historical data: {e}.")
    df_historical['DATE_TIME'] = pd.to_datetime(df_historical['DATE_TIME'], errors='coerce')
    if df_historical['DATE_TIME'].isnull().any():
        print(f"Warning: Some DATE_TIME values in historical data are NaT. Dropping them.")
        df_historical.dropna(subset=['DATE_TIME'], inplace=True)

df_historical = df_historical.set_index('DATE_TIME').sort_index()
df_historical = df_historical.drop_duplicates()

cols_to_clean_and_convert = USER_PROVIDED_FUTURE_COLS + MODEL_PREDICTED_COLS + ['DAILY_YIELD', 'TOTAL_YIELD']
for col in cols_to_clean_and_convert:
    if col in df_historical.columns:
        if df_historical[col].isnull().any():
             df_historical[col] = df_historical[col].ffill().bfill()
        df_historical[col] = pd.to_numeric(df_historical[col], errors='coerce').fillna(0)
    else:
        print(f"Warning: Column '{col}' not found in historical DataFrame during cleaning.")

df_historical = df_historical.dropna(subset=['PLANT_ID', 'SOURCE_KEY'])
print(f"Shape after cleaning historical data: {df_historical.shape}")
if df_historical.empty: exit("Historical DataFrame is empty after preprocessing.")


# --- 3. Feature Engineering Function ---
print("\n--- 3. Feature Engineering ---")
def create_features_for_prediction(dataf_input, is_future_data=False):
    df_feat = dataf_input.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['month'] = df_feat.index.month

    minutes_in_day = 24 * 60
    df_feat['timeofday_sin'] = np.sin(2 * np.pi * (df_feat['hour'] * 60 + df_feat['minute']) / minutes_in_day)
    df_feat['timeofday_cos'] = np.cos(2 * np.pi * (df_feat['hour'] * 60 + df_feat['minute']) / minutes_in_day)
    df_feat['dayofyear_sin'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365.0)
    df_feat['dayofyear_cos'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365.0)

    if 'MODULE_TEMPERATURE' in df_feat.columns and 'IRRADIATION' in df_feat.columns:
        df_feat['TEMP_X_IRRADIATION'] = df_feat['MODULE_TEMPERATURE'] * df_feat['IRRADIATION']
    
    # For historical data, create lags of all relevant columns for training
    # For future data, user provides weather, so we only create lags of those *provided* weather values
    # Lags for power in future data are more complex (recursive) and are omitted here for simplicity,
    # meaning the model primarily relies on current weather and time features for future power.
    cols_for_lags = USER_PROVIDED_FUTURE_COLS
    if not is_future_data: # For historical data, include power columns for lag generation
        cols_for_lags = cols_for_lags + MODEL_PREDICTED_COLS

    for col in cols_for_lags:
        if col in df_feat.columns:
            df_feat[f'{col}_lag_{RECORDS_PER_HOUR_FORECAST}'] = df_feat[col].shift(RECORDS_PER_HOUR_FORECAST)
            df_feat[f'{col}_lag_{RECORDS_PER_HOUR_FORECAST*24}'] = df_feat[col].shift(RECORDS_PER_HOUR_FORECAST*24)
            df_feat[f'{col}_roll_mean_{RECORDS_PER_HOUR_FORECAST*3}'] = df_feat[col].rolling(window=RECORDS_PER_HOUR_FORECAST*3, min_periods=1).mean()

    lag_roll_cols = [c for c in df_feat.columns if '_lag_' in c or '_roll_mean_' in c]
    if lag_roll_cols:
        df_feat[lag_roll_cols] = df_feat[lag_roll_cols].bfill().ffill()
        df_feat[lag_roll_cols] = df_feat[lag_roll_cols].fillna(0)
    return df_feat

df_historical_featured = create_features_for_prediction(df_historical, is_future_data=False)

df_historical_featured['PLANT_ID'] = df_historical_featured['PLANT_ID'].astype(str)
df_historical_featured['SOURCE_KEY'] = df_historical_featured['SOURCE_KEY'].astype(str)
df_historical_featured = pd.get_dummies(df_historical_featured, columns=['PLANT_ID', 'SOURCE_KEY'], prefix=['PLANT', 'SOURCE'], dummy_na=False)
print(f"Shape of historical data with features: {df_historical_featured.shape}")


# --- Helper function for model training ---
def train_power_model(df_train_data, target_col, feature_cols_list):
    print(f"Training model for: {target_col}")
    
    existing_feature_cols = [f_col for f_col in feature_cols_list if f_col in df_train_data.columns]
    if not existing_feature_cols:
        print(f"  Error: No features available for {target_col}. Skipping training.")
        return None, None, []

    X_train = df_train_data[existing_feature_cols].copy() # Use copy
    y_train = df_train_data[target_col].copy()

    if X_train.empty or y_train.empty or len(X_train) < 20:
        print(f"  Not enough data to train model for {target_col}. Skipping.")
        return None, None, []

    scaler = StandardScaler()
    numeric_features_to_scale = X_train.select_dtypes(include=np.number).columns.tolist()
    ohe_prefixes = ('PLANT_', 'SOURCE_')
    time_sin_cos_suffixes = ('_sin', '_cos')
    scalable_cols_specific = [
        c for c in numeric_features_to_scale 
        if not c.startswith(ohe_prefixes) and \
           not c.endswith(time_sin_cos_suffixes) and \
           X_train[c].nunique(dropna=False) > 2 # Consider NaNs if any slipped
    ]

    X_train_scaled = X_train.copy()
    if scalable_cols_specific:
        # Ensure columns to scale actually exist in X_train_scaled before trying to transform
        valid_scalable_cols = [col for col in scalable_cols_specific if col in X_train_scaled.columns]
        if valid_scalable_cols:
            X_train_scaled[valid_scalable_cols] = scaler.fit_transform(X_train[valid_scalable_cols])
        else:
            print(f"  Warning: No valid columns found to scale for {target_col} from scalable_cols_specific list.")
    
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, 
                                  max_depth=25, min_samples_split=8, min_samples_leaf=4, oob_score=True)
    model.fit(X_train_scaled, y_train) # Pass X_train_scaled
    if hasattr(model, 'oob_score_') and model.oob_score_ :
        print(f"  OOB Score for {target_col}: {model.oob_score_:.4f}")
    return model, scaler, scalable_cols_specific


# --- 4. Train Models for DC_POWER and AC_POWER ---
print("\n--- 4. Training Power Models (on all historical data) ---")
models = {}
scalers = {}
scalable_columns_map = {}

base_plus_cyclical_ohe_features = [
    'hour', 'minute', 'dayofweek', 'dayofyear', 'month',
    'timeofday_sin', 'timeofday_cos', 'dayofyear_sin', 'dayofyear_cos'
]
base_plus_cyclical_ohe_features.extend([col for col in df_historical_featured.columns if col.startswith('PLANT_') or col.startswith('SOURCE_')])

current_target_dc = 'DC_POWER'
features_for_dc = base_plus_cyclical_ohe_features[:] + USER_PROVIDED_FUTURE_COLS
for col in USER_PROVIDED_FUTURE_COLS: 
    features_for_dc.extend([f_name for f_name in [f'{col}_lag_{RECORDS_PER_HOUR_FORECAST}', f'{col}_lag_{RECORDS_PER_HOUR_FORECAST*24}', f'{col}_roll_mean_{RECORDS_PER_HOUR_FORECAST*3}'] if f_name in df_historical_featured.columns])
if 'TEMP_X_IRRADIATION' in df_historical_featured.columns: features_for_dc.append('TEMP_X_IRRADIATION')
for f_name in [f'{current_target_dc}_lag_{RECORDS_PER_HOUR_FORECAST}', f'{current_target_dc}_lag_{RECORDS_PER_HOUR_FORECAST*24}', f'{current_target_dc}_roll_mean_{RECORDS_PER_HOUR_FORECAST*3}']:
    if f_name in df_historical_featured.columns: features_for_dc.append(f_name)

if current_target_dc in df_historical_featured.columns:
    models[current_target_dc], scalers[current_target_dc], scalable_columns_map[current_target_dc] = \
        train_power_model(df_historical_featured, current_target_dc, list(set(features_for_dc)))

current_target_ac = 'AC_POWER'
features_for_ac = base_plus_cyclical_ohe_features[:] + [current_target_dc] + USER_PROVIDED_FUTURE_COLS
for col in USER_PROVIDED_FUTURE_COLS + [current_target_dc]: 
    features_for_ac.extend([f_name for f_name in [f'{col}_lag_{RECORDS_PER_HOUR_FORECAST}', f'{col}_lag_{RECORDS_PER_HOUR_FORECAST*24}', f'{col}_roll_mean_{RECORDS_PER_HOUR_FORECAST*3}'] if f_name in df_historical_featured.columns])
if 'TEMP_X_IRRADIATION' in df_historical_featured.columns: features_for_ac.append('TEMP_X_IRRADIATION')
for f_name in [f'{current_target_ac}_lag_{RECORDS_PER_HOUR_FORECAST}', f'{current_target_ac}_lag_{RECORDS_PER_HOUR_FORECAST*24}', f'{current_target_ac}_roll_mean_{RECORDS_PER_HOUR_FORECAST*3}']:
    if f_name in df_historical_featured.columns: features_for_ac.append(f_name)

if current_target_ac in df_historical_featured.columns:
    models[current_target_ac], scalers[current_target_ac], scalable_columns_map[current_target_ac] = \
        train_power_model(df_historical_featured, current_target_ac, list(set(features_for_ac)))

print(f"\nPower models trained: {list(models.keys())}")


# --- 5. Function to Get Future Weather Input from User ---
def get_future_weather_from_user(start_dt_for_future, num_periods, interval_minutes):
    print(f"\n--- User Input for Future Weather Required ---")
    print(f"Please provide weather data for {num_periods} intervals ({N_HOURS_TO_FORECAST} hours at {interval_minutes} min intervals).")
    
    future_weather_df = pd.DataFrame(index=pd.date_range(start=start_dt_for_future, periods=num_periods, freq=f"{interval_minutes}T"))

    file_path_input = input(f"Enter path to CSV file with future weather data OR press Enter to use simulated data (persistence): ").strip()

    if file_path_input and os.path.exists(file_path_input):
        print(f"Loading future weather data from: {file_path_input}")
        try:
            user_df = pd.read_csv(file_path_input)
            user_df['DATE_TIME'] = pd.to_datetime(user_df['DATE_TIME'])
            user_df = user_df.set_index('DATE_TIME')

            # Align and fill
            future_weather_df_temp = user_df.reindex(future_weather_df.index) # Align to our expected timestamps
            
            for col_name in USER_PROVIDED_FUTURE_COLS:
                if col_name in future_weather_df_temp.columns:
                    future_weather_df[col_name] = pd.to_numeric(future_weather_df_temp[col_name], errors='coerce')
                    if future_weather_df[col_name].isnull().any():
                        print(f"  NaNs found in user-provided '{col_name}'. Attempting ffill/bfill.")
                        future_weather_df[col_name] = future_weather_df[col_name].ffill().bfill()
                    if future_weather_df[col_name].isnull().any(): # Still NaNs after fill
                         raise ValueError(f"Column '{col_name}' in user CSV has unfillable NaNs for the forecast period.")
                else:
                    raise ValueError(f"Required column '{col_name}' not found in user-provided CSV.")
            print("Successfully loaded and processed user-provided future weather data.")
            return future_weather_df[USER_PROVIDED_FUTURE_COLS]

        except Exception as e:
            print(f"Error processing user-provided CSV: {e}")
            print("Falling back to simulated data (persistence).")
            # Fallback to persistence is handled below if this path fails
    else:
        if file_path_input: # User entered a path but it doesn't exist
            print(f"File not found at '{file_path_input}'.")
        print("No valid CSV provided or path incorrect. Using simulated data (persistence from 24h ago).")

    # Fallback: Simulate data using persistence if no valid file provided
    for col_name in USER_PROVIDED_FUTURE_COLS:
        future_weather_df[col_name] = 0 
        for f_ts_val in future_weather_df.index:
            past_ts = f_ts_val - pd.Timedelta(days=1)
            try:
                val_idx = df_historical.index.get_indexer([past_ts], method='nearest')[0]
                val_persist = df_historical.iloc[val_idx][col_name] if val_idx != -1 else df_historical[col_name].iloc[-1]
                future_weather_df.loc[f_ts_val, col_name] = val_persist
            except Exception:
                future_weather_df.loc[f_ts_val, col_name] = df_historical[col_name].iloc[-1] if not df_historical.empty and col_name in df_historical else 0
        print(f"  Simulated future {col_name} (using persistence from 24h ago).")
    return future_weather_df[USER_PROVIDED_FUTURE_COLS]


# --- 6. Forecasting Power for Next N_HOURS_TO_FORECAST ---
print(f"\n--- 6. Forecasting Power for Next {N_HOURS_TO_FORECAST} Hours ---")

last_known_dt_hist = df_historical_featured.index.max()
data_freq_obj = pd.Timedelta(minutes=FORECAST_INTERVAL_MINUTES)
start_dt_for_future = last_known_dt_hist + data_freq_obj

future_df_input_weather = get_future_weather_from_user(start_dt_for_future, NUM_FUTURE_PERIODS, FORECAST_INTERVAL_MINUTES)

# Combine future weather with OHE and then generate all features
future_df_for_features = future_df_input_weather.copy()
ohe_cols = [col for col in df_historical_featured.columns if col.startswith('PLANT_') or col.startswith('SOURCE_')]
if not df_historical_featured.empty and ohe_cols:
    last_ohe_values = df_historical_featured[ohe_cols].iloc[-1]
    for ohe_col, val_ohe in last_ohe_values.items():
        future_df_for_features[ohe_col] = val_ohe
else:
    example_plant_col_name = next((c for c in df_historical_featured.columns if c.startswith("PLANT_")), None)
    example_source_col_name = next((c for c in df_historical_featured.columns if c.startswith("SOURCE_")), None)
    if example_plant_col_name: future_df_for_features[example_plant_col_name] = 1
    if example_source_col_name: future_df_for_features[example_source_col_name] = 1

future_df_featured = create_features_for_prediction(future_df_for_features, is_future_data=True)

# --- Predict DC_POWER ---
target_col_dc = 'DC_POWER'
print(f"Forecasting: {target_col_dc}")
if target_col_dc in models and models[target_col_dc] is not None:
    model_dc = models[target_col_dc]
    scaler_dc = scalers[target_col_dc]
    scalable_cols_dc = scalable_columns_map[target_col_dc]
    features_dc_model = model_dc.feature_names_in_

    X_future_dc_prep = future_df_featured.copy()
    if 'TEMP_X_IRRADIATION' in features_dc_model and \
       'MODULE_TEMPERATURE' in X_future_dc_prep.columns and \
       'IRRADIATION' in X_future_dc_prep.columns:
        X_future_dc_prep['TEMP_X_IRRADIATION'] = X_future_dc_prep['MODULE_TEMPERATURE'] * X_future_dc_prep['IRRADIATION']
    for f_name in features_dc_model:
        if f_name not in X_future_dc_prep.columns: X_future_dc_prep[f_name] = 0 # Add missing features as 0
    
    X_future_dc = X_future_dc_prep[features_dc_model] # Select features in correct order
    X_future_dc_scaled = X_future_dc.copy()

    if scaler_dc and scalable_cols_dc:
        cols_to_scale_now = [c for c in scalable_cols_dc if c in X_future_dc_scaled.columns]
        if cols_to_scale_now:
            X_future_dc_scaled[cols_to_scale_now] = scaler_dc.transform(X_future_dc[cols_to_scale_now])
    
    dc_predictions = model_dc.predict(X_future_dc_scaled)
    future_df_featured[target_col_dc] = np.maximum(0, dc_predictions)
else:
    future_df_featured[target_col_dc] = 0

# --- Predict AC_POWER ---
target_col_ac = 'AC_POWER'
print(f"Forecasting: {target_col_ac}")
if target_col_ac in models and models[target_col_ac] is not None:
    model_ac = models[target_col_ac]
    scaler_ac = scalers[target_col_ac]
    scalable_cols_ac = scalable_columns_map[target_col_ac]
    features_ac_model = model_ac.feature_names_in_
    
    X_future_ac_prep = future_df_featured.copy() # Now contains predicted DC_POWER
    if 'TEMP_X_IRRADIATION' in features_ac_model and \
       'MODULE_TEMPERATURE' in X_future_ac_prep.columns and \
       'IRRADIATION' in X_future_ac_prep.columns:
        X_future_ac_prep['TEMP_X_IRRADIATION'] = X_future_ac_prep['MODULE_TEMPERATURE'] * X_future_ac_prep['IRRADIATION']
    for f_name in features_ac_model:
        if f_name not in X_future_ac_prep.columns: X_future_ac_prep[f_name] = 0
            
    X_future_ac = X_future_ac_prep[features_ac_model]
    X_future_ac_scaled = X_future_ac.copy()

    if scaler_ac and scalable_cols_ac:
        cols_to_scale_now = [c for c in scalable_cols_ac if c in X_future_ac_scaled.columns]
        if cols_to_scale_now:
            X_future_ac_scaled[cols_to_scale_now] = scaler_ac.transform(X_future_ac[cols_to_scale_now])
            
    ac_predictions = model_ac.predict(X_future_ac_scaled)
    future_df_featured[target_col_ac] = np.maximum(0, ac_predictions)
else:
    future_df_featured[target_col_ac] = 0

# --- Calculate Yields ---
# (Yield calculation logic remains the same as your previous correct version)
print("\nCalculating DAILY_YIELD and TOTAL_YIELD from forecasted AC_POWER...")
if 'AC_POWER' in future_df_featured.columns:
    energy_per_interval_kwh = future_df_featured['AC_POWER'] * (FORECAST_INTERVAL_MINUTES / 60.0)
    last_total_yield_historical = df_historical['TOTAL_YIELD'].iloc[-1] if not df_historical.empty else 0
    future_df_featured['TOTAL_YIELD_forecasted'] = last_total_yield_historical + energy_per_interval_kwh.cumsum()

    last_daily_yield_historical = 0
    last_daily_yield_date_historical = None
    if not df_historical.empty:
        last_hist_dt_series = df_historical.loc[df_historical.index == last_known_dt_hist]
        if not last_hist_dt_series.empty:
            last_daily_yield_historical_record = last_hist_dt_series.iloc[0]
            last_daily_yield_historical = last_daily_yield_historical_record['DAILY_YIELD']
            last_daily_yield_date_historical = last_daily_yield_historical_record.name.date()
        elif not df_historical.empty : 
             last_daily_yield_historical = df_historical['DAILY_YIELD'].iloc[-1]
             last_daily_yield_date_historical = df_historical.index[-1].date()

    future_df_featured['DAILY_YIELD_forecasted'] = 0.0
    current_day_yield = 0.0
    if not future_df_featured.empty and last_daily_yield_date_historical is not None:
        if future_df_featured.index[0].date() == last_daily_yield_date_historical:
            current_day_yield = last_daily_yield_historical
    
    previous_date = future_df_featured.index[0].date() if not future_df_featured.empty else None
    for timestamp, energy in energy_per_interval_kwh.items():
        current_date = timestamp.date()
        if current_date != previous_date:
            if timestamp != future_df_featured.index[0] or current_day_yield == 0 :
                 current_day_yield = 0.0
        current_day_yield += energy
        future_df_featured.loc[timestamp, 'DAILY_YIELD_forecasted'] = current_day_yield
        previous_date = current_date
else:
    future_df_featured['TOTAL_YIELD_forecasted'] = 0
    future_df_featured['DAILY_YIELD_forecasted'] = 0


# --- Output to Terminal and Plot ---
cols_to_output = MODEL_PREDICTED_COLS + DERIVED_COLS # DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD

print("\n--- Forecasted Power and Yield Values (Next 24 Hours) ---")
# Ensure all columns exist before trying to print
existing_output_cols = [col for col in cols_to_output if col in future_df_featured.columns]
if existing_output_cols:
    print(future_df_featured[existing_output_cols].to_string())
else:
    print("No columns available to display in the forecast output.")


print("\n--- 7. Plotting Forecasts ---")
# For plotting, include the user-provided weather as well
plot_display_cols = USER_PROVIDED_FUTURE_COLS + MODEL_PREDICTED_COLS + DERIVED_COLS
num_plots = len(plot_display_cols)
fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3 * num_plots), sharex=True) # Reduced height per plot
if num_plots == 0:
    print("No data to plot.")
elif num_plots == 1: 
    axes = [axes]

for i, col_to_plot in enumerate(plot_display_cols):
    ax = axes[i]
    historical_col_name = col_to_plot.replace('_forecasted', '')
    
    if historical_col_name in df_historical_featured.columns:
        history_to_plot_len = min(len(df_historical_featured), RECORDS_PER_HOUR_FORECAST * 24 * 2)
        historical_data_plot = df_historical_featured[historical_col_name].iloc[-history_to_plot_len:]
        ax.plot(historical_data_plot.index, historical_data_plot.values, label=f'Historical {historical_col_name}', alpha=0.7)
    
    if col_to_plot in future_df_featured.columns:
        label_prefix = "Forecasted"
        if col_to_plot in USER_PROVIDED_FUTURE_COLS: label_prefix = "User-Provided"
        elif col_to_plot in DERIVED_COLS: label_prefix = "Derived"
        ax.plot(future_df_featured.index, future_df_featured[col_to_plot], label=f'{label_prefix} {col_to_plot}', linestyle='--')
    
    ax.set_ylabel(col_to_plot); ax.legend(); ax.grid(True)

if num_plots > 0:
    axes[-1].set_xlabel('Date Time')
    fig.suptitle(f'Solar Forecast (User Weather) - Next {N_HOURS_TO_FORECAST} Hours', fontsize=16)
    plt.xticks(rotation=45); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

print("\n--- Script Finished ---") 