import io
import time
import cv2
import numpy as np
import logging
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
MODEL_PATH = 'best.pt'
BAD_RICE_LABELS = ['Damaged', 'Discolored', 'Broken', 'Chalky', 'Organic Foreign Matters']

# Load YOLO model
try:
    model_load_start_time = time.time()
    model = YOLO(MODEL_PATH)
    model_load_end_time = time.time()
    print(f"YOLO model '{MODEL_PATH}' loaded successfully in {model_load_end_time - model_load_start_time:.2f} seconds.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure your model path is correct and ultralytics is installed.")
    exit()

@app.route('/')
def index():
    return "Welcome to the Rice Quality Detection API! Send a POST request to /predict with an image."

@app.route('/predict', methods=['POST'])
def predict():
    if 'image/jpeg' not in request.content_type:
        return jsonify({"error": "Unsupported Media Type. Expected image/jpeg"}), 415

    reception_start_time = time.time()
    image_data = request.data
    reception_end_time = time.time()

    # Decode image
    try:
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    # Run inference
    inference_start_time = time.time()
    results = model(img)
    inference_end_time = time.time()

    detections = []
    bad_rice_detected = False
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls_id)]
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(conf),
                "label": label
            })
            if label in BAD_RICE_LABELS:
                bad_rice_detected = True

    print(f"Image reception and decoding time: {reception_end_time - reception_start_time:.4f} seconds")
    print(f"Model inference time: {inference_end_time - inference_start_time:.4f} seconds")
    print(f"Total processing time: {inference_end_time - reception_start_time:.4f} seconds")
    print(f"Detected {len(detections)} objects. Bad rice detected: {bad_rice_detected}")
    print("-" * 50)

    return jsonify({"detections": detections, "bad_rice_detected": bad_rice_detected}), 200

# ✅ ADDED THIS FUNCTION so pipx can run it as a CLI entry point
def main():
    from gunicorn.app.wsgiapp import run
    run()

# ✅ COMMENTED THIS OUT because pipx won't call '__main__' directly
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
