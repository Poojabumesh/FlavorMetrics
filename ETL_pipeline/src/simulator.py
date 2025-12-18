import json, os, random, time
from datetime import datetime
import paho.mqtt.client as mqtt

BROKER_HOST = os.getenv("BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))

TOPIC = os.getenv("MQTT_TOPIC", "factory/beer/sensors")
BATCH_LIFETIME_HOURS = float(os.getenv("BATCH_LIFETIME_HOURS", "3"))
BATCH_LIFETIME_SECONDS = BATCH_LIFETIME_HOURS * 3600

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
# Track creation time so we can rotate IDs after a set lifetime.
batch_ids = {}  # (plant_id, line_id) -> (batch_id, created_ts)

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def get_batch_id(plant_id: str, line_id: str) -> str:
    key = (plant_id, line_id)
    now_ts = time.time()
    batch, created = batch_ids.get(key, (None, None))
    if batch is None or (now_ts - created) >= BATCH_LIFETIME_SECONDS:
        batch = f"batch-{plant_id}-{line_id}-{int(now_ts)}"
        batch_ids[key] = (batch, now_ts)
        print(f"[simulator] rotated batch for {plant_id}/{line_id} → {batch}")
    return batch

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
    # four signals: temp, pH, gravity (degP), CO2 pressure
    temp_std = PLANTS[plant_id]["fermentation_temp_std"]
    temp = random.gauss(20, temp_std)
    ph = random.gauss(4.2, 0.15)
    gravity_degP = random.gauss(12.0, 0.6)
    co2_pressure = random.gauss(0.9, 0.08)  # bar
    return [
        {
            "step": "fermentation",
            "sensor": "temp",
            "value": round(temp, 2),
            "unit": "C",
        },
        {
            "step": "fermentation",
            "sensor": "ph",
            "value": round(ph, 2),
            "unit": "pH",
        },
        {
            "step": "fermentation",
            "sensor": "gravity_degP",
            "value": round(gravity_degP, 2),
            "unit": "degP",
        },
        {
            "step": "fermentation",
            "sensor": "co2_pressure",
            "value": round(co2_pressure, 3),
            "unit": "bar",
        },
    ]


def gen_packaging(line_id: str, plant_id: str):
    # count bottles / min + quality signals
    low, high = PLANTS[plant_id]["lines"][line_id]["packaging_range"]
    count = random.randint(low, high)
    fill_level = random.gauss(335, 4)  # ml target 330-340
    cap_torque = random.gauss(16, 1.8)  # N·cm
    line_speed = random.gauss((low + high) / 2, 8)
    reject_rate = max(0.0, random.gauss(1.5, 0.6))  # percent
    return [
        {
            "step": "packaging",
            "sensor": "count",
            "value": count,
            "unit": "units",
        },
        {
            "step": "packaging",
            "sensor": "fill_level",
            "value": round(fill_level, 2),
            "unit": "ml",
        },
        {
            "step": "packaging",
            "sensor": "cap_torque",
            "value": round(cap_torque, 2),
            "unit": "Ncm",
        },
        {
            "step": "packaging",
            "sensor": "line_speed",
            "value": round(line_speed, 1),
            "unit": "units/min",
        },
        {
            "step": "packaging",
            "sensor": "reject_rate",
            "value": round(reject_rate, 2),
            "unit": "%",
        },
    ]

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
