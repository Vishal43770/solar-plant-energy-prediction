import pandas as pd

# File paths
gen_path = "/home/vishal/office/databricks/datasets/Plant_2_Generation_Data.csv"
weather_path = "/home/vishal/office/databricks/datasets/Plant_2_Weather_Sensor_Data.csv"

# Read generation data with required columns
gen_cols = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']
gen_df = pd.read_csv(gen_path, usecols=gen_cols)

# Read weather data with required columns
weather_cols = ['DATE_TIME', 'PLANT_ID', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
weather_df = pd.read_csv(weather_path, usecols=weather_cols)

# Ensure consistent datetime format
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], dayfirst=True)
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], dayfirst=True)

# Merge on DATE_TIME and PLANT_ID
combined_df = pd.merge(gen_df, weather_df, on=['DATE_TIME', 'PLANT_ID'], how='left')

# Optional: save the merged file
output_path = "/home/vishal/office/databricks/datasets/Combined_Generation_Weather_Data2.csv"
combined_df.to_csv(output_path, index=False)

print(f"✅ Combined dataset saved to: {output_path}")
