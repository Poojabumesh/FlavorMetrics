import json, os, random, time
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

BROKER_HOST = os.getenv("BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))

TOPIC = os.getenv("MQTT_TOPIC", "factory/beer/sensors")

client = mqtt.Client()
client.connect(BROKER_HOST, BROKER_PORT, 60)

batch_id = f"batch-{int(time.time())}"
plant_id = "plantA"
line_id = "line1"

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def gen_mashing():
    val = random.gauss(65, 1.5)
    return {
        "step" : "mashing",
        "sensor" : "temp",
        "value" : round(val, 2),
        "unit" : "C",
        }

def gen_boiling():
    val = random.gauss(99, 0.7)
    return {
        "step" : "boiling",
        "sensor" : "temp",
        "value" : round(val, 2),
        "unit" : "C",
        }

def gen_fermentation():
    # two signals: temp + gravity
    temp = random.gauss(20, 0.5)
    gravity = random.gauss(1.015, 0.003)
    return [
        {
            "step": "fermentation",
            "sensor": "temp",
            "value": round(temp, 2),
            "unit": "C",
        },
        {
            "step": "fermentation",
            "sensor": "gravity",
            "value": round(gravity, 4),
            "unit": "SG",
        },
    ]


def gen_packaging():
    # count bottles / min
    count = random.randint(90, 110)
    return {
        "step": "packaging",
        "sensor": "count",
        "value": count,
        "unit": "units",
    }

generators = [gen_mashing, gen_boiling, gen_fermentation, gen_packaging]


print(f"publishing beer batch {batch_id} ... Ctrl+C to stop")
while True:
    g = random.choice(generators)
    data = g()
    if isinstance(data, list):
        msgs = data
    else:
        msgs = [data]

    for m in msgs:
        payload = {
            "plant_id": plant_id,
            "line_id": line_id,
            "batch_id": batch_id,
            "ts": now_iso(),
            **m,
        }
        client.publish(TOPIC, json.dumps(payload))
    time.sleep(1.0)
