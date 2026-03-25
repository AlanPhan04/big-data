import time
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
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

# Collect data to simulate a streaming input
all_data = df_prepared.select("vehicle", "latitude", "longitude", "prev_lat", "prev_lon", "event_epoch").collect()

split_idx = int(len(all_data) * 0.9) # Use 90% for training, 10% for simulated streaming
train_batch_rows = all_data[:split_idx]
stream_mock_rows = all_data[split_idx:]

LAT_BASE, LON_BASE = 10.0, 106.0

# Function to convert data into ML format (using new ML Vectors)
def to_ml_df(rows, target_col, feature_col, offset):
    data = []
    for r in rows:
        label = float(r[target_col]) - offset
        # KNN-like behavior: only one feature which is the previous coordinate
        features = Vectors.dense([float(r[feature_col]) - offset])
        data.append((label, features))
    return spark.createDataFrame(data, ["label", "features"])

# =========================================================
# 2. INITIALIZE DECISION TREE (OPTIMIZED PARAMETERS)
# =========================================================
# maxDepth=10: Deep enough to capture turns but not too computationally heavy
# maxBins=32: Number of split candidates for continuous features
dt_lat = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=10, maxBins=32)
dt_lon = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=10, maxBins=32)

print(f">>> Training decision tree on {len(train_batch_rows)} points...")

df_train_lat = to_ml_df(train_batch_rows, "latitude", "prev_lat", LAT_BASE)
df_train_lon = to_ml_df(train_batch_rows, "longitude", "prev_lon", LON_BASE)

model_lat = dt_lat.fit(df_train_lat)
model_lon = dt_lon.fit(df_train_lon)

# =========================================================
# 3. HAVERSINE & STREAMING SIMULATION
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

print("\n>>> START REAL-TIME PREDICTION (ONLINE PREDICTION)...")

# Sliding window of the most recent 500 points for retraining
window_data = train_batch_rows[-500:]

try:
    for i, row in enumerate(stream_mock_rows):
        # A. PREDICT (TEST)
        feat_lat = Vectors.dense([row['prev_lat'] - LAT_BASE])
        feat_lon = Vectors.dense([row['prev_lon'] - LON_BASE])
        
        p_lat = float(model_lat.predict(feat_lat)) + LAT_BASE
        p_lon = float(model_lon.predict(feat_lon)) + LON_BASE
        
        err_km = haversine(p_lat, p_lon, row['latitude'], row['longitude'])
        
        print(f"!"*40)
        print(f"[{i+1}/{len(stream_mock_rows)}] Vehicle: {row['vehicle']} | Time: {row['event_epoch']}")
        print(f" -> Predicted: ({p_lat:.6f}, {p_lon:.6f})")
        print(f" -> Actual: ({row['latitude']:.6f}, {row['longitude']:.6f})")
        print(f" -> Error: {err_km*1000:.2f} meters") # Convert to meters for readability

        # B. UPDATE MODEL (RETRAIN EVERY 10 POINTS)
        window_data.append(row)
        window_data.pop(0)
        
        if (i + 1) % 10 == 0:
            df_up_lat = to_ml_df(window_data, "latitude", "prev_lat", LAT_BASE)
            df_up_lon = to_ml_df(window_data, "longitude", "prev_lon", LON_BASE)
            
            model_lat = dt_lat.fit(df_up_lat)
            model_lon = dt_lon.fit(df_up_lon)
            print(f">>> [Update] Model knowledge updated using the last 10 new points.")

        time.sleep(0.5) # Run slightly faster

except KeyboardInterrupt:
    print("\nPrediction stream stopped.")
finally:
    spark.stop()