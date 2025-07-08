import time
import cv2
import numpy as np
import logging
from flask import Flask, request, jsonify
from ultralytics import YOLO

# ✅ Flask app instance
app = Flask(__name__)

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ Model configuration
MODEL_PATH = 'best.pt'
BAD_RICE_LABELS = ['Damaged', 'Discolored', 'Broken', 'Chalky', 'Organic Foreign Matters']

# ✅ Load YOLO model safely
try:
    start = time.time()
    model = YOLO(MODEL_PATH)
    duration = time.time() - start
    print(f"✅ YOLO model '{MODEL_PATH}' loaded in {duration:.2f} seconds.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# ✅ Home route
@app.route('/')
def index():
    return "🌾 Rice Quality Detection API – POST a JPEG image to /predict."

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image/jpeg' not in request.content_type:
        return jsonify({"error": "Expected image/jpeg content type"}), 415

    try:
        raw = request.data
        np_img = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decoding failed")
    except Exception as e:
        logger.error(f"Decoding error: {e}")
        return jsonify({"error": "Invalid image format"}), 400

    # ✅ Run detection
    start = time.time()
    results = model(img)
    duration = time.time() - start

    detections = []
    bad_rice = False

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        ids = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(score),
                "label": label
            })
            if label in BAD_RICE_LABELS:
                bad_rice = True

    print(f"🧠 Inference time: {duration:.3f}s | Detected: {len(detections)} | Bad rice: {bad_rice}")
    return jsonify({"detections": detections, "bad_rice_detected": bad_rice}), 200
