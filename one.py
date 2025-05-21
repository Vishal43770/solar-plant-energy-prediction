from pyspark.sql import SparkSession

 # Create a SparkSession
spark = SparkSession.builder.appName("Csv Reader").getOrCreate()

 # Read the CSV files into DataFrames
df1 = spark.read.csv("/home/vishal/office/databricks/datasets/Combined_Generation_Weather_Data1.csv",
                  header=True, inferSchema=True)

df2 = spark.read.csv("/home/vishal/office/databricks/datasets/Combined_Generation_Weather_Data2.csv",
                  header=True, inferSchema=True)

 # Print the DataFrames
print("Data from Combined_Generation_Weather_Data1.csv")
df1.show()
print("\n\nData from Combined_Generation_Weather_Data2.csv")
df2.show()

df1_selected = df1.select("DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE")
df2_selected = df2.select("DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE")

# # Print the selected DataFrames
print("\n\nData from Combined_Generation_Weather_Data1.csv after selecting columns")
df1_selected.show()

print("\n\nData from Combined_Generation_Weather_Data2.csv after selecting columns")
df2_selected.show()

# Stop the SparkSession
spark.stop()