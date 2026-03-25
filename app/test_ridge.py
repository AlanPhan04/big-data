import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.streaming import StreamingContext
from pyspark.sql.types import *
import math

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

split_idx = int(len(all_data) * 0.7)
train_batch_rows = all_data[:split_idx]
stream_mock_rows = all_data[split_idx:]

LAT_BASE, LON_BASE = 10.0, 106.0

# Helper function to convert data into Spark ML format (Vector)
def to_ml_df(rows, target_col, feature_col, offset):
    data = []
    for r in rows:
        # Normalize data using an offset
        label = float(r[target_col]) - offset
        features = Vectors.dense([float(r[feature_col]) - offset])
        data.append((label, features))
    return spark.createDataFrame(data, ["label", "features"])

# =========================================================
# 2. INITIALIZE & TRAIN RIDGE MODEL (70%)
# =========================================================
# regParam=0.1 (Ridge coefficient), elasticNetParam=0.0 (pure Ridge)
ridge_lat = LinearRegression(regParam=0.1, elasticNetParam=0.0, maxIter=10)
ridge_lon = LinearRegression(regParam=0.1, elasticNetParam=0.0, maxIter=10)

print(f">>> Training Ridge Model on {len(train_batch_rows)} points...")

df_train_lat = to_ml_df(train_batch_rows, "latitude", "prev_lat", LAT_BASE)
df_train_lon = to_ml_df(train_batch_rows, "longitude", "prev_lon", LON_BASE)

model_lat = ridge_lat.fit(df_train_lat)
model_lon = ridge_lon.fit(df_train_lon)

# =========================================================
# 3. HAVERSINE FUNCTION & MANUAL STREAMING SIMULATION
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

print(">>> Starting streaming simulation (Rolling Ridge learning)...")

# Sliding window data used for retraining the model (keep the latest 100 points)
window_data_lat = train_batch_rows[-100:] 
window_data_lon = train_batch_rows[-100:]

try:
    for i, row in enumerate(stream_mock_rows):
        # A. PREDICT (Test)
        feat_lat = Vectors.dense([row['prev_lat'] - LAT_BASE])
        feat_lon = Vectors.dense([row['prev_lon'] - LON_BASE])
        
        # Use the current model to make predictions
        p_lat = float(model_lat.predict(feat_lat)) + LAT_BASE
        p_lon = float(model_lon.predict(feat_lon)) + LON_BASE
        
        err_km = haversine(p_lat, p_lon, row['latitude'], row['longitude'])
        
        print(f"\n[{i+1}/{len(stream_mock_rows)}] VEHICLE: {row['vehicle']}")
        print(f" -> Predicted: ({p_lat:.6f}, {p_lon:.6f})")
        print(f" -> Actual: ({row['latitude']:.6f}, {row['longitude']:.6f})")
        print(f" -> Error: {err_km:.10f} km")

        # B. UPDATE MODEL (Manual training)
        # Add new point to sliding window and remove the oldest point
        window_data_lat.append(row)
        window_data_lon.append(row)
        window_data_lat.pop(0)
        window_data_lon.pop(0)
        
        # Retrain the Ridge model using the most recent data (online learning simulation)
        if (i + 1) % 5 == 0: # Update model weights every 5 points to reduce CPU usage
            df_up_lat = to_ml_df(window_data_lat, "latitude", "prev_lat", LAT_BASE)
            df_up_lon = to_ml_df(window_data_lon, "longitude", "prev_lon", LON_BASE)
            model_lat = ridge_lat.fit(df_up_lat)
            model_lon = ridge_lon.fit(df_up_lon)
            print("--- Ridge Model weights updated ---")

        time.sleep(1) # Simulate delay between GPS pings

except KeyboardInterrupt:
    print("\nSimulation finished.")