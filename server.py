import io
import time
import cv2
import numpy as np
import logging
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# 📌 Improved logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔧 Configuration
MODEL_PATH = 'best.pt'
BAD_RICE_LABELS = ['Damaged', 'Discolored', 'Broken', 'Chalky', 'Organic Foreign Matters']

# 🔒 Safe model loading
try:
    model_load_start = time.time()
    model = YOLO(MODEL_PATH)
    load_duration = time.time() - model_load_start
    print(f"✅ YOLO model loaded in {load_duration:.2f} seconds")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit()

@app.route('/')
def home():
    return "🌾 Rice Quality Detection API – POST an image to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image/jpeg' not in request.content_type:
        return jsonify({"error": "Unsupported Media Type. Expected image/jpeg"}), 415

    reception_start = time.time()
    img_bytes = request.data
    reception_end = time.time()

    try:
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decoding failed.")
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    infer_start = time.time()
    results = model(img)
    infer_end = time.time()

    detections = []
    bad_rice_found = False

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy()

        for box, score, label_id in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(label_id)]
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(score),
                "label": label
            })
            if label in BAD_RICE_LABELS:
                bad_rice_found = True

    print(f"📥 Reception & decoding time: {reception_end - reception_start:.4f}s")
    print(f"🧠 Inference time: {infer_end - infer_start:.4f}s")
    print(f"⏱️ Total time: {infer_end - reception_start:.4f}s")
    print(f"🧾 Detections: {len(detections)}, Bad rice: {bad_rice_found}")

    return jsonify({
        "detections": detections,
        "bad_rice_detected": bad_rice_found
    }), 200

# 🔑 Entry point for pipx CLI
def main():
    from gunicorn.app.wsgiapp import run
    run()
