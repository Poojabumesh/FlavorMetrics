import json, os, random, time
from datetime import datetime
import paho.mqtt.client as mqtt

BROKER_HOST = os.getenv("BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))

TOPIC = os.getenv("MQTT_TOPIC", "factory/beer/sensors")

client = mqtt.Client()
client.connect(BROKER_HOST, BROKER_PORT, 60)

PLANTS = {
    "plantA": {
        "fermentation_temp_std": 1.25,  # intentionally more variable
        "lines": {
            "line1": {"mode": "bottling", "packaging_range": (90, 115)},
            "line2": {"mode": "canning", "packaging_range": (120, 150)},
        },
    },
    "plantB": {
        "fermentation_temp_std": 0.55,
        "lines": {
            "line1": {"mode": "bottling", "packaging_range": (95, 120)},
            "line2": {"mode": "canning", "packaging_range": (125, 155)},
        },
    },
}

# Keep a batch id per (plant,line) so messages look like concurrent batches.
batch_ids = {}

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def get_batch_id(plant_id: str, line_id: str) -> str:
    key = (plant_id, line_id)
    if key not in batch_ids:
        batch_ids[key] = f"batch-{plant_id}-{line_id}-{int(time.time())}"
    return batch_ids[key]

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

def gen_fermentation(plant_id: str):
    # two signals: temp + gravity
    temp_std = PLANTS[plant_id]["fermentation_temp_std"]
    temp = random.gauss(20, temp_std)
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


def gen_packaging(line_id: str, plant_id: str):
    # count bottles / min
    low, high = PLANTS[plant_id]["lines"][line_id]["packaging_range"]
    count = random.randint(low, high)
    return {
        "step": "packaging",
        "sensor": "count",
        "value": count,
        "unit": "units",
    }

generators = ["mashing", "boiling", "fermentation", "packaging"]


print("publishing beer batches for plants ['plantA', 'plantB'] ... Ctrl+C to stop")
while True:
    plant_id = random.choice(list(PLANTS.keys()))
    line_id = random.choice(list(PLANTS[plant_id]["lines"].keys()))
    batch_id = get_batch_id(plant_id, line_id)

    step = random.choice(generators)
    if step == "mashing":
        data = gen_mashing()
    elif step == "boiling":
        data = gen_boiling()
    elif step == "fermentation":
        data = gen_fermentation(plant_id)
    else:
        data = gen_packaging(line_id, plant_id)

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
