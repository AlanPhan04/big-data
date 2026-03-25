import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.mllib.regression import LabeledPoint, StreamingLinearRegressionWithSGD
from pyspark.mllib.linalg import Vectors
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

all_data = df_prepared.select("vehicle", "latitude", "longitude", "prev_lat", "prev_lon", "event_epoch").collect()

split_idx = int(len(all_data) * 0.7)
train_batch = all_data[:split_idx]
stream_mock = all_data[split_idx:]

LAT_BASE = 10.0
LON_BASE = 106.0

model_lat = StreamingLinearRegressionWithSGD(stepSize=0.01, numIterations=10)
model_lon = StreamingLinearRegressionWithSGD(stepSize=0.01, numIterations=10)

model_lat.setInitialWeights([1.0])
model_lon.setInitialWeights([1.0])

if train_batch:
    rdd_warmup = sc.parallelize(train_batch)
    
    lp_lat = rdd_warmup.map(lambda r: LabeledPoint(r.latitude - LAT_BASE, [r.prev_lat - LAT_BASE]))
    lp_lon = rdd_warmup.map(lambda r: LabeledPoint(r.longitude - LON_BASE, [r.prev_lon - LON_BASE]))

    model_lat.trainOn(ssc.queueStream([lp_lat]))
    model_lon.trainOn(ssc.queueStream([lp_lon]))
    print(f">>> Warm-up complete with {len(train_batch)} GPS location points.")


input_queue = [sc.parallelize(stream_mock[:1])]
stream = ssc.queueStream(input_queue, oneAtATime=True)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c

def process_and_compare(rdd):
    if not rdd.isEmpty():
        items = rdd.collect()
        for row in items:
            try:
                d = row.asDict() if hasattr(row, "asDict") else row
                
                f_lat = Vectors.dense([d['prev_lat'] - LAT_BASE])
                f_lon = Vectors.dense([d['prev_lon'] - LON_BASE])
                
                p_lat = model_lat.latestModel().predict(f_lat) + LAT_BASE
                p_lon = model_lon.latestModel().predict(f_lon) + LON_BASE
                
                actual_lat = d['latitude']
                actual_lon = d['longitude']

                dist_error = haversine(p_lat, p_lon, actual_lat, actual_lon)
                
                print("\n" + "!"*50, flush=True)
                print(f" Vehicle ID: {d.get('vehicle', 'N/A')} | Unix Timestamp: {d.get('event_epoch')}")
                print(f" -> Latitude: Predicted {p_lat:.6f} | Actual {actual_lat:.6f}")
                print(f" -> Longtitude: Predicted {p_lon:.6f} | Actual {actual_lon:.6f}")
                print(f" -> Difference (km): {dist_error:.10f}")
                print("!"*50 + "\n", flush=True)
            except Exception as e:
                print(f"Error processing: {e}")

stream.foreachRDD(process_and_compare)

model_lat.trainOn(stream.map(lambda r: LabeledPoint(r.latitude - LAT_BASE, [r.prev_lat - LAT_BASE])))
model_lon.trainOn(stream.map(lambda r: LabeledPoint(r.longitude - LON_BASE, [r.prev_lon - LON_BASE])))

if len(stream_mock) > 0:
    model_lat._stepSize = 0.05
    model_lon._stepSize = 0.05
    
    ssc.start()
    print(">>> Begin streaming simulation...")
    time.sleep(2)

    try:
        for i, record in enumerate(stream_mock):
            current_rdd = sc.parallelize([record])
            input_queue.append(current_rdd)
            print(f"--- Pushed {i+1}/{len(stream_mock)} into the Queue ---", flush=True)
            process_and_compare(current_rdd)
            time.sleep(2) 
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        ssc.stop(stopSparkContext=True, stopGracefully=True)