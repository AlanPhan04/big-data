import time
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from pyspark.streaming import StreamingContext

spark = SparkSession.builder.appName("GPS-Univariate-Simulation").getOrCreate()
sc = spark.sparkContext
ssc = StreamingContext(sc, 2)
ssc.remember(10)

spark.sparkContext.setLogLevel("ERROR")

schema = StructType([
    StructField("msgType", StringType(), True),
    StructField("msgBusWayPoint", StructType([
        StructField("vehicle", StringType(), True),
        StructField("datetime", LongType(), True),
        StructField("speed", DoubleType(), True),
        StructField("x", DoubleType(), True),
        StructField("y", DoubleType(), True),
    ]), True)
])

input_path = "/archive/*/*/*.json"
df_raw = spark.read.option("multiline", "true").schema(schema).json(input_path)

df_flat = df_raw.select(
    col("msgBusWayPoint.vehicle").alias("vehicle"),
    col("msgBusWayPoint.datetime").alias("event_epoch"),
    col("msgBusWayPoint.x").alias("longitude"),
    col("msgBusWayPoint.y").alias("latitude"),
)

window_spec = Window.partitionBy("vehicle").orderBy("event_epoch")
df_prepared = df_flat.withColumn("prev_lat", lag("latitude").over(window_spec)) \
                     .withColumn("prev_lon", lag("longitude").over(window_spec)) \
                     .dropna()

# Collect data to the Driver to simulate a streaming input
all_data = df_prepared.select("vehicle", "latitude", "longitude", "prev_lat", "prev_lon", "event_epoch").collect()

split_idx = int(len(all_data) * 0.9)
train_batch_rows = all_data[:split_idx]
stream_mock_rows = all_data[split_idx:]

LAT_BASE, LON_BASE = 10.0, 106.0

def to_ml_df(rows, target_col, feature_col, offset):
    data = []
    for r in rows:
        label = float(r[target_col]) - offset
        features = Vectors.dense([float(r[feature_col]) - offset])
        data.append((label, features))
    return spark.createDataFrame(data, ["label", "features"])

# =========================================================
# 2. INITIALIZE ELASTIC NET
# =========================================================
# regParam: Strength of the regularization penalty
# elasticNetParam: 0.5 means a balanced mix between L1 and L2 regularization
en_lat = LinearRegression(regParam=0.01, elasticNetParam=0.5, maxIter=20)
en_lon = LinearRegression(regParam=0.01, elasticNetParam=0.5, maxIter=20)

print(f">>> Warming up Elastic Net with {len(train_batch_rows)} points...")

df_train_lat = to_ml_df(train_batch_rows, "latitude", "prev_lat", LAT_BASE)
df_train_lon = to_ml_df(train_batch_rows, "longitude", "prev_lon", LON_BASE)

model_lat = en_lat.fit(df_train_lat)
model_lon = en_lon.fit(df_train_lon)

# =========================================================
# 3. HAVERSINE & STREAMING SIMULATION
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

print("\n>>> START PREDICTION WITH ELASTIC NET...")

# Sliding window used to update knowledge (e.g., keep the most recent 1000 points)
window_data = train_batch_rows[-1000:]

try:
    for i, row in enumerate(stream_mock_rows):
        # A. PREDICT
        feat_lat = Vectors.dense([row['prev_lat'] - LAT_BASE])
        feat_lon = Vectors.dense([row['prev_lon'] - LON_BASE])
        
        p_lat = float(model_lat.predict(feat_lat)) + LAT_BASE
        p_lon = float(model_lon.predict(feat_lon)) + LON_BASE
        
        err_m = haversine(p_lat, p_lon, row['latitude'], row['longitude']) * 1000
        
        print("-" * 30)
        print(f"Batch: {i+1} | Vehicle: {row['vehicle']}")
        print(f" -> Predicted: ({p_lat:.6f}, {p_lon:.6f})")
        print(f" -> Actual: ({row['latitude']:.6f}, {row['longitude']:.6f})")
        print(f" -> Error: {err_m:.2f} meters")

        # B. UPDATE (Rolling / incremental learning)
        window_data.append(row)
        window_data.pop(0)
        
        # Periodically update the model (e.g., every 20 points)
        if (i + 1) % 20 == 0:
            df_up_lat = to_ml_df(window_data, "latitude", "prev_lat", LAT_BASE)
            df_up_lon = to_ml_df(window_data, "longitude", "prev_lon", LON_BASE)
            model_lat = en_lat.fit(df_up_lat)
            model_lon = en_lon.fit(df_up_lon)
            print(">>> [Elastic Net] Model weights have been retrained using new data.")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nFinished.")
finally:
    spark.stop()