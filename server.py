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
    print("Please ensure your model path is correct and ultralytics is installed.")
    exit() # Exit if model cannot be loaded

# ✅ Home route
@app.route('/')
def index():
    return "🌾 Rice Quality Detection API – POST a JPEG image to /predict."

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image/jpeg' not in request.content_type:
        return jsonify({"error": "Expected image/jpeg content type"}), 415

    reception_start_time = time.time()
    image_data = request.data # Get raw image bytes
    reception_end_time = time.time()

    # Convert image data to OpenCV format
    try:
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    # ✅ Run detection
    start = time.time()
    results = model(img)
    duration = time.time() - start
    end = time.time()

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
    print(f"Image reception and decoding time: {reception_end_time - reception_start_time:.4f} seconds")
    print(f"Model inference time: {duration:.4f} seconds")
    print(f"Total processing time on server: {end - reception_start_time:.4f} seconds")
    print(f"Detected {len(detections)} objects. Bad rice detected: {bad_rice}")
    print("-" * 50)
    return jsonify({"detections": detections, "bad_rice_detected": bad_rice}), 200
