from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, from_unixtime, to_timestamp

# =========================================================
# 1. CREATE SPARK SESSION
# =========================================================
spark = (
    SparkSession.builder
        .appName("BusWayPoint-ETL")
        .config("spark.sql.shuffle.partitions", "200")
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

# =========================================================
# 4. FLATTEN STRUCT
# =========================================================
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

print(df_flat.show(10))

# # =========================================================
# # 5. CONVERT EPOCH → TIMESTAMP
# # =========================================================
# df_silver = (
#     df_flat
#         .withColumn(
#             "event_time",
#             to_timestamp(from_unixtime(col("event_epoch")))
#         )
#         .drop("event_epoch")
# )

# # =========================================================
# # 6. FILTER ONLY BUS WAYPOINT (OPTIONAL)
# # =========================================================
# df_silver = df_silver.filter(col("msgType") == "MsgType_BusWayPoint")

# # =========================================================
# # 7. REPARTITION (AVOID SMALL FILE PROBLEM)
# # =========================================================
# df_silver = df_silver.repartition(200)

# # =========================================================
# # 8. WRITE TO PARQUET (BRONZE → SILVER)
# # =========================================================
# output_path = "/lakehouse/silver/bus_waypoint"

# (
#     df_silver.write
#         .mode("overwrite")      # change to append if incremental
#         .format("parquet")
#         .save(output_path)
# )

print("===== DONE =====")

spark.stop()

#  /opt/spark/bin/spark-submit test_sparkml.py 