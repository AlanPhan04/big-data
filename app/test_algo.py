from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, row_number, lead, max as spark_max, lag
)
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# =========================================================
# 1. CREATE SPARK SESSION
# =========================================================
spark = (
    SparkSession.builder
    .appName("GPS-Ridge-Streaming-Simulation")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# =========================================================
# 2. DEFINE SCHEMA (NO INFER FOR PERFORMANCE)
# =========================================================
schema = StructType([
    StructField("msgType", StringType(), True),
    StructField("msgBusWayPoint", StructType([
        StructField("vehicle", StringType(), True),
        StructField("driver", StringType(), True),
        StructField("datetime", LongType(), True),
        StructField("speed", DoubleType(), True),
        StructField("x", DoubleType(), True),
        StructField("y", DoubleType(), True),
        StructField("ignition", BooleanType(), True),
        StructField("aircon", BooleanType(), True),
    ]), True)
])

# =========================================================
# 3. READ ALL JSON FILES (RECURSIVE)
# =========================================================
input_path = "/archive/*/*/*.json"
print(1)
df_raw = (
    spark.read
        .option("multiline", "true")  # IMPORTANT (JSON array)
        .schema(schema)
        .json(input_path)
)

print("===== RAW SCHEMA =====")
df_raw.printSchema()



df_flat = df_raw.select(
    col("msgType"),
    col("msgBusWayPoint.vehicle").alias("vehicle"),
    col("msgBusWayPoint.driver").alias("driver"),
    col("msgBusWayPoint.datetime").alias("event_epoch"),
    col("msgBusWayPoint.speed").alias("speed"),
    col("msgBusWayPoint.x").alias("longitude"),
    col("msgBusWayPoint.y").alias("latitude"),
    col("msgBusWayPoint.ignition").alias("ignition"),
    col("msgBusWayPoint.aircon").alias("aircon"),
)


df = df_flat.select(
    "vehicle",
    "event_epoch",
    "speed",
    "longitude",
    "latitude",
)

# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna()

df = df.withColumn("event_epoch", col("event_epoch").cast("long"))
df = df.withColumn("speed", col("speed").cast("double"))
df = df.withColumn("longitude", col("longitude").cast("double"))
df = df.withColumn("latitude", col("latitude").cast("double"))

# =========================================================
# PARTITION theo vehicle để tránh OOM
# =========================================================

window_spec = Window.partitionBy("vehicle").orderBy("event_epoch")

df = df.withColumn("prev_lat", lag("latitude").over(window_spec))
df = df.withColumn("prev_lon", lag("longitude").over(window_spec))
df = df.withColumn("prev_speed", lag("speed").over(window_spec))

df = df.dropna()

# =========================================================
# SAMPLE DATA (QUAN TRỌNG)
# =========================================================

# dataset 103M -> chỉ lấy 1% để train
df = df.sample(fraction=0.01, seed=42)

# =========================================================
# FEATURES
# =========================================================

assembler = VectorAssembler(
    inputCols=["prev_lat", "prev_lon", "prev_speed"],
    outputCol="features"
)

df = assembler.transform(df)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

print("Train count:", train_df.count())
print("Test count:", test_df.count())

# =========================================================
# TRAIN MODEL
# =========================================================

lr_lat = LinearRegression(
    featuresCol="features",
    labelCol="latitude",
    maxIter=10
)

lr_lon = LinearRegression(
    featuresCol="features",
    labelCol="longitude",
    maxIter=10
)

model_lat = lr_lat.fit(train_df)
model_lon = lr_lon.fit(train_df)

# =========================================================
# PREDICT
# =========================================================

pred_lat = model_lat.transform(test_df)

pred_lat = pred_lat.withColumnRenamed(
    "prediction",
    "predicted_latitude"
)

pred_lon = model_lon.transform(pred_lat)

pred_lon = pred_lon.withColumnRenamed(
    "prediction",
    "predicted_longitude"
)

# =========================================================
# SAVE RESULT TXT
# =========================================================

result = pred_lon.select(
    "vehicle",
    "event_epoch",
    "latitude",
    "longitude",
    "predicted_latitude",
    "predicted_longitude"
)

print(result.show())
# result.coalesce(1).write.mode("overwrite").text("prediction_output")

print("Done!")

spark.stop()