import io
import time
import cv2
import numpy as np
import socket
import os
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

def get_server_ip():
    """Get the server's IP address"""
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def display_server_info(port):
    """Display server information on startup"""
    server_ip = get_server_ip()
    print("=" * 60)
    print(f"🚀 ESP-CAM Rice Detection Server Started!")
    print(f"📡 Server IP: {server_ip}")
    print(f"🔌 Port: {port}")
    print(f"🌐 Full URL: http://{server_ip}:{port}")
    print(f"📋 API Endpoint: http://{server_ip}:{port}/predict")
    print(f"💻 Local URL: http://localhost:{port}/predict")
    print("=" * 60)

# =========================================================
# !!! IMPORTANT !!!
# Replace 'best.pt' with the path to your trained model file.
MODEL_PATH = 'best.pt'

# !!! IMPORTANT !!!
# Define a list of all class names that should trigger the relay.
# Ensure these labels exactly match the output class names from your YOLO model.
BAD_RICE_LABELS = ['Damaged', 'Discolored', 'Broken', 'Chalky', 'Organic Foreign Matters']
# =========================================================

# Load the YOLO modelpyh
try:
    model_load_start_time = time.time()
    model = YOLO(MODEL_PATH)
    model_load_end_time = time.time()
    # Corrected typo here: changed 'model_load_end_end_time' to 'model_load_end_time'
    print(f"YOLO model '{MODEL_PATH}' loaded successfully in {model_load_end_time - model_load_start_time:.2f} seconds.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure your model path is correct and ultralytics is installed.")
    exit() # Exit if model cannot be loaded

@app.route('/predict', methods=['POST'])
def predict():
    if 'image/jpeg' not in request.content_type:
        return jsonify({"error": "Unsupported Media Type. Expected image/jpeg"}), 415

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

    # Perform inference
    inference_start_time = time.time()
    results = model(img) # Predict on the image
    inference_end_time = time.time()

    # Process results
    detections = []
    bad_rice_detected = False # Flag to indicate if any bad rice is found
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = r.boxes.conf.cpu().numpy() # Confidence scores
        class_ids = r.boxes.cls.cpu().numpy()   # Class IDs

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls_id)] # Get class name from model
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(conf),
                "label": label
            })
            # Check if the detected label is in our list of "bad" rice labels
            if label in BAD_RICE_LABELS:
                bad_rice_detected = True # Set flag to true if any bad rice is found

    print(f"Image reception and decoding time: {reception_end_time - reception_start_time:.4f} seconds")
    print(f"Model inference time: {inference_end_time - inference_start_time:.4f} seconds")
    print(f"Total processing time on server: {inference_end_time - reception_start_time:.4f} seconds")
    print(f"Detected {len(detections)} objects. Bad rice detected: {bad_rice_detected}")
    print("-" * 50)

    # Include the bad_rice_detected flag in the JSON response
    return jsonify({"detections": detections, "bad_rice_detected": bad_rice_detected}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Display server information
    display_server_info(port)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
