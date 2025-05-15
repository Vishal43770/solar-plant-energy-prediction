import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_PATH1 = '/home/vishal/office/databricks/datasets/Combined_Generation_Weather_Data1.csv'
DATA_PATH2 = '/home/vishal/office/databricks/datasets/Combined_Generation_Weather_Data2.csv'

# Columns we want to predict
TARGET_COLUMNS_TO_PREDICT = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_POWER', 'AC_POWER']

N_HOURS_TEST = 48  # Hours for historical test set
N_HOURS_TO_FORECAST = 24
FORECAST_INTERVAL_MINUTES = 15

# --- 1. Load and Combine Data ---
print("--- 1. Loading and Combining Data ---")
try:
    df1 = pd.read_csv(DATA_PATH1)
    df2 = pd.read_csv(DATA_PATH2)
    df = pd.concat([df1, df2], ignore_index=True)
except FileNotFoundError:
    print(f"Error: CSV files not found.")
    exit()
print(f"Shape of combined data: {df.shape}")

# --- 2. Initial Exploration & Cleaning ---
print("\n--- 2. Initial Exploration & Cleaning ---")
try:
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
except Exception as e:
    print(f"Error parsing DATE_TIME: {e}. Attempting common formats.")
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], errors='coerce')
    if df['DATE_TIME'].isnull().any():
        print(f"Warning: Some DATE_TIME values are NaT. Dropping them.")
        df.dropna(subset=['DATE_TIME'], inplace=True)

df = df.set_index('DATE_TIME').sort_index()
df = df.drop_duplicates()

# NaN Handling & Type Conversion
all_relevant_cols = TARGET_COLUMNS_TO_PREDICT + ['DAILY_YIELD', 'TOTAL_YIELD'] # Include yields for cleaning
for col in all_relevant_cols:
    if col in df.columns:
        if df[col].isnull().any(): # Only ffill/bfill if NaNs exist
             df[col] = df[col].ffill().bfill() # Fill then backfill
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Convert to numeric, fill remaining NaNs with 0
    else:
        print(f"Warning: Column '{col}' not found in DataFrame during cleaning.")


df = df.dropna(subset=['PLANT_ID', 'SOURCE_KEY']) # Drop if these identifiers are NaN
print(f"Shape after cleaning: {df.shape}")
if df.empty: exit("DataFrame is empty after preprocessing.")

# --- 3. Feature Engineering (Common Time Features) ---
print("\n--- 3. Feature Engineering ---")
def create_base_features(dataf):
    df_feat = dataf.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute # For 15-min intervals
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['month'] = df_feat.index.month
    # Add more as needed: 'weekofyear', 'quarter'
    return df_feat

df_featured = create_base_features(df)

# Handle PLANT_ID and SOURCE_KEY (used as features for all models)
df_featured['PLANT_ID'] = df_featured['PLANT_ID'].astype(str)
df_featured['SOURCE_KEY'] = df_featured['SOURCE_KEY'].astype(str)
df_featured = pd.get_dummies(df_featured, columns=['PLANT_ID', 'SOURCE_KEY'], prefix=['PLANT', 'SOURCE'], dummy_na=False)
print(f"Shape with base features: {df_featured.shape}")


# --- Helper function for model training ---
def train_single_target_model(df_train_data, target_col, feature_cols_list):
    print(f"Training model for: {target_col}")
    
    X_train = df_train_data[feature_cols_list]
    y_train = df_train_data[target_col]

    # Simple check for sufficient data
    if X_train.empty or y_train.empty or len(X_train) < 10:
        print(f"Not enough data to train model for {target_col}. Skipping.")
        return None, None

    scaler = StandardScaler()
    # Identify numeric columns in X_train for scaling (excluding OHE columns)
    numeric_features_to_scale = X_train.select_dtypes(include=np.number).columns.tolist()
    ohe_prefixes = ('PLANT_', 'SOURCE_') # Columns from get_dummies
    scalable_cols_specific = [c for c in numeric_features_to_scale if not c.startswith(ohe_prefixes) and X_train[c].nunique() > 2]

    X_train_scaled = X_train.copy()
    if scalable_cols_specific:
        X_train_scaled[scalable_cols_specific] = scaler.fit_transform(X_train[scalable_cols_specific])
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=10)
    model.fit(X_train_scaled, y_train)
    return model, scaler, scalable_cols_specific


# --- Determine data frequency for future predictions ---
# Assuming 15-minute interval as requested for forecasting
data_frequency_for_range = pd.Timedelta(minutes=FORECAST_INTERVAL_MINUTES)
records_per_hour_forecast = 60 // FORECAST_INTERVAL_MINUTES


# --- 4. Train Models for Each Target ---
print("\n--- 4. Training Separate Models ---")
models = {}
scalers = {}
scalable_columns_map = {}

# Define base features (time features + OHE plant/source)
base_feature_names = [col for col in df_featured.columns if col.startswith('hour') or \
                      col.startswith('minute') or col.startswith('dayofweek') or \
                      col.startswith('dayofyear') or col.startswith('month') or \
                      col.startswith('PLANT_') or col.startswith('SOURCE_')]

# Train AMBIENT_TEMPERATURE model
# Features: time features
current_target = 'AMBIENT_TEMPERATURE'
features_for_ambient = base_feature_names[:] # Use a copy
if current_target in df_featured.columns and not df_featured[features_for_ambient].empty:
    models[current_target], scalers[current_target], scalable_columns_map[current_target] = train_single_target_model(df_featured, current_target, features_for_ambient)

# Train IRRADIATION model
# Features: time features
current_target = 'IRRADIATION'
features_for_irradiation = base_feature_names[:]
if current_target in df_featured.columns and not df_featured[features_for_irradiation].empty:
    models[current_target], scalers[current_target], scalable_columns_map[current_target] = train_single_target_model(df_featured, current_target, features_for_irradiation)

# Train MODULE_TEMPERATURE model
# Features: time features, AMBIENT_TEMPERATURE, IRRADIATION
current_target = 'MODULE_TEMPERATURE'
features_for_module = base_feature_names[:] + ['AMBIENT_TEMPERATURE', 'IRRADIATION']
features_for_module = [f for f in features_for_module if f in df_featured.columns] # Ensure all features exist
if current_target in df_featured.columns and features_for_module and not df_featured[features_for_module].empty:
    models[current_target], scalers[current_target], scalable_columns_map[current_target] = train_single_target_model(df_featured, current_target, features_for_module)

# Train DC_POWER model
# Features: time features, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION
current_target = 'DC_POWER'
features_for_dc = base_feature_names[:] + ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
features_for_dc = [f for f in features_for_dc if f in df_featured.columns]
if current_target in df_featured.columns and features_for_dc and not df_featured[features_for_dc].empty:
    models[current_target], scalers[current_target], scalable_columns_map[current_target] = train_single_target_model(df_featured, current_target, features_for_dc)

# Train AC_POWER model
# Features: time features, DC_POWER (or AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION)
current_target = 'AC_POWER'
features_for_ac = base_feature_names[:] + ['DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION'] # Using more features
# features_for_ac = base_feature_names[:] + ['DC_POWER'] # Simpler version if DC_POWER is predicted well
features_for_ac = [f for f in features_for_ac if f in df_featured.columns]
if current_target in df_featured.columns and features_for_ac and not df_featured[features_for_ac].empty:
    models[current_target], scalers[current_target], scalable_columns_map[current_target] = train_single_target_model(df_featured, current_target, features_for_ac)

print(f"\nModels trained: {list(models.keys())}")
# ... (Keep all code from before Section 5 as is) ...

# --- 5. Forecasting Next N_HOURS_TO_FORECAST ---
print(f"\n--- 5. Forecasting Next {N_HOURS_TO_FORECAST} Hours ---")

last_known_dt = df_featured.index.max()
num_future_periods = N_HOURS_TO_FORECAST * records_per_hour_forecast

start_dt_for_future_range = last_known_dt + data_frequency_for_range
future_datetimes = pd.date_range(start=start_dt_for_future_range, periods=num_future_periods, freq=data_frequency_for_range)
future_df = pd.DataFrame(index=future_datetimes)
print(f"Predicting for {num_future_periods} future 15-min intervals.")

# Populate base features for future_df
future_df = create_base_features(future_df)

# Add OHE PLANT_ID and SOURCE_KEY (assuming prediction for the last known plant/source)
ohe_cols = [col for col in df_featured.columns if col.startswith('PLANT_') or col.startswith('SOURCE_')]
if not df_featured.empty and ohe_cols:
    last_ohe_values = df_featured[ohe_cols].iloc[-1]
    for ohe_col, val in last_ohe_values.items():
        future_df[ohe_col] = val
else:
    print("Warning: Could not get OHE columns from historical data for future predictions.")
    example_plant_col = [c for c in base_feature_names if c.startswith("PLANT_")]
    example_source_col = [c for c in base_feature_names if c.startswith("SOURCE_")]
    if example_plant_col: future_df[example_plant_col[0]] = 1
    if example_source_col: future_df[example_source_col[0]] = 1

# Iterative Forecasting Loop
for target_col in TARGET_COLUMNS_TO_PREDICT: # This list does NOT include DAILY_YIELD or TOTAL_YIELD
    print(f"Forecasting: {target_col}")
    if target_col not in models or models[target_col] is None:
        print(f"  No model for {target_col}, skipping its prediction or using persistence.")
        if target_col in ['AMBIENT_TEMPERATURE', 'IRRADIATION', 'MODULE_TEMPERATURE']:
            # Simple persistence for weather if model failed/missing
            for f_idx, f_ts_val in enumerate(future_df.index): # Iterate using index and value
                past_ts = f_ts_val - pd.Timedelta(days=1)
                try:
                    # Find nearest data point in historical data
                    val_idx = df.index.get_indexer([past_ts], method='nearest')[0]
                    val = df.iloc[val_idx][target_col] if val_idx != -1 else df[target_col].iloc[-1]
                    future_df.loc[f_ts_val, target_col] = val
                except Exception as e_persist:
                    # print(f"    Persistence error for {target_col} at {f_ts_val}: {e_persist}")
                    future_df.loc[f_ts_val, target_col] = df[target_col].iloc[-1] # Fallback to very last known
            print(f"  Used persistence for {target_col}.")
        else:
            future_df[target_col] = 0 # Default to 0 for power if no model
        continue

    model = models[target_col]
    scaler = scalers[target_col]
    scalable_cols_for_target = scalable_columns_map[target_col]
    
    if hasattr(model, 'feature_names_in_'):
        features_for_current_model = model.feature_names_in_
    else: 
        if target_col == 'AMBIENT_TEMPERATURE': features_for_current_model = features_for_ambient
        elif target_col == 'IRRADIATION': features_for_current_model = features_for_irradiation
        elif target_col == 'MODULE_TEMPERATURE': features_for_current_model = features_for_module
        elif target_col == 'DC_POWER': features_for_current_model = features_for_dc
        elif target_col == 'AC_POWER': features_for_current_model = features_for_ac
        else: features_for_current_model = base_feature_names 
    
    missing_model_features = [f for f in features_for_current_model if f not in future_df.columns]
    if missing_model_features:
        print(f"  Warning: Model for {target_col} needs features {missing_model_features} not yet in future_df. Setting to 0.")
        for mf in missing_model_features:
            future_df[mf] = 0 

    X_future = future_df[features_for_current_model].copy()
    X_future_scaled = X_future.copy()

    if scaler and scalable_cols_for_target:
        cols_to_scale_now = [c for c in scalable_cols_for_target if c in X_future_scaled.columns]
        if cols_to_scale_now:
             X_future_scaled[cols_to_scale_now] = scaler.transform(X_future[cols_to_scale_now])
    
    predictions = model.predict(X_future_scaled)
    
    if target_col in ['IRRADIATION', 'DC_POWER', 'AC_POWER']:
        predictions = np.maximum(0, predictions)
        
    future_df[target_col] = predictions

# --- Calculate DAILY_YIELD and TOTAL_YIELD from predicted AC_POWER ---
print("\nCalculating DAILY_YIELD and TOTAL_YIELD from forecasted AC_POWER...")
if 'AC_POWER' in future_df.columns:
    # AC_POWER is in kW, interval is 15 minutes (0.25 hours)
    # Energy per interval (kWh) = AC_POWER (kW) * (15/60 hours)
    energy_per_interval_kwh = future_df['AC_POWER'] * (FORECAST_INTERVAL_MINUTES / 60.0)

    # Calculate TOTAL_YIELD
    last_total_yield_historical = df['TOTAL_YIELD'].iloc[-1] if not df.empty else 0
    future_df['TOTAL_YIELD_forecasted'] = last_total_yield_historical + energy_per_interval_kwh.cumsum()

    # Calculate DAILY_YIELD
    # Needs to reset each day. Find the last known DAILY_YIELD and its date.
    last_daily_yield_historical = 0
    last_daily_yield_date_historical = None

    if not df.empty:
        # Find last non-zero daily yield to understand typical end-of-day values if needed,
        # but more importantly, the last recorded daily yield on the *last day of historical data*.
        last_hist_dt_series = df.loc[df.index == last_known_dt] # Get all records for the last known timestamp
        if not last_hist_dt_series.empty:
            # If multiple plants/inverters, this could be tricky. Assume we take the max or mean,
            # or ideally, this should be per plant if your original data supports it clearly.
            # For simplicity, taking the value from the first entry at last_known_dt.
            last_daily_yield_historical_record = last_hist_dt_series.iloc[0]
            last_daily_yield_historical = last_daily_yield_historical_record['DAILY_YIELD']
            last_daily_yield_date_historical = last_daily_yield_historical_record.name.date() # Get the date part
        else: # Fallback if last_known_dt somehow not in df
             last_daily_yield_historical = df['DAILY_YIELD'].iloc[-1]
             last_daily_yield_date_historical = df.index[-1].date()


    future_df['DAILY_YIELD_forecasted'] = 0.0
    current_day_yield = 0.0
    
    # Determine the starting yield for the first forecast day
    if not future_df.empty and last_daily_yield_date_historical is not None:
        if future_df.index[0].date() == last_daily_yield_date_historical:
            # Forecast starts on the same day as last historical record
            current_day_yield = last_daily_yield_historical
        # Else, forecast starts on a new day, so current_day_yield remains 0 initially
    
    previous_date = future_df.index[0].date() if not future_df.empty else None

    for timestamp, energy in energy_per_interval_kwh.items():
        current_date = timestamp.date()
        if current_date != previous_date:
            # Day has changed, reset daily yield accumulator (unless it's the very first iteration and we carried over)
            if timestamp != future_df.index[0] or current_day_yield == 0 : # if not first entry or if first entry is for a new day
                 current_day_yield = 0.0
        
        current_day_yield += energy
        future_df.loc[timestamp, 'DAILY_YIELD_forecasted'] = current_day_yield
        previous_date = current_date
else:
    print("  AC_POWER not predicted, cannot calculate forecasted yields.")
    future_df['TOTAL_YIELD_forecasted'] = 0
    future_df['DAILY_YIELD_forecasted'] = 0

# Add the newly calculated yields to the list of columns to display/plot
display_and_plot_cols = TARGET_COLUMNS_TO_PREDICT[:]
if 'DAILY_YIELD_forecasted' in future_df: display_and_plot_cols.append('DAILY_YIELD_forecasted')
if 'TOTAL_YIELD_forecasted' in future_df: display_and_plot_cols.append('TOTAL_YIELD_forecasted')


print("\n--- Forecasted Values (Next 24 Hours, including calculated Yields) ---")
print(future_df[display_and_plot_cols].head())
print(future_df[display_and_plot_cols].describe())


# --- 6. Plotting Results ---
print("\n--- 6. Plotting Forecasts ---")
num_plots = len(display_and_plot_cols) # Now includes yields
fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4 * num_plots), sharex=True) # Adjusted height per plot
if num_plots == 1: axes = [axes]

for i, target_col_to_plot in enumerate(display_and_plot_cols):
    ax = axes[i]
    
    # Determine the historical column name (original name, not '_forecasted')
    historical_col_name = target_col_to_plot.replace('_forecasted', '')
    
    # Plot some history for context
    if historical_col_name in df_featured.columns:
        history_to_plot_len = min(len(df_featured), records_per_hour_forecast * 24 * 3) # Approx last 3 days
        historical_data = df_featured[historical_col_name].iloc[-history_to_plot_len:]
        ax.plot(historical_data.index, historical_data.values, label=f'Historical {historical_col_name}', alpha=0.7)
    
    # Plot forecast
    if target_col_to_plot in future_df.columns:
        ax.plot(future_df.index, future_df[target_col_to_plot], label=f'Forecasted {target_col_to_plot}', linestyle='--')
    
    ax.set_ylabel(target_col_to_plot)
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel('Date Time')
fig.suptitle(f'Multivariate Forecast for Next {N_HOURS_TO_FORECAST} Hours ({FORECAST_INTERVAL_MINUTES}-min intervals)', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n--- Script Finished ---")

# ... (rest of the script, if any)