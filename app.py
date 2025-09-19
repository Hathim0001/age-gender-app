# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import cv2
import numpy as np
import insightface
import threading

app = Flask(__name__)
CORS(app)  # allow cross-origin if needed (adjust in production)

# Load InsightFace model once (heavy). Protect inference with a lock for thread-safety.
print("Loading InsightFace model (this may take a while the first run)...")
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))
model_lock = threading.Lock()
print("Model ready.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expect JSON body: { "image": "data:image/jpeg;base64,..." }
    Returns: { "detections": [ {bbox:[x1,y1,x2,y2], age:int, gender:"Male"/"Female"} ], "width":w, "height":h }
    """
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({"error": "no image provided"}), 400

    img_b64 = data['image']
    # remove data URI prefix if present
    if img_b64.startswith('data:image'):
        img_b64 = img_b64.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR image
        if img is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        return jsonify({"error": "invalid image", "detail": str(e)}), 400

    orig_h, orig_w = img.shape[:2]

    # optional: scale input down to speed up inference if very large
    max_side = 800
    scale = 1.0
    if max(orig_h, orig_w) > max_side:
        scale = max_side / max(orig_h, orig_w)
        img = cv2.resize(img, (int(orig_w*scale), int(orig_h*scale)))

    # inference (thread-safe)
    with model_lock:
        faces = model.get(img)

    detections = []
    for face in faces:
        # face.bbox is [x1, y1, x2, y2] as float64
        bbox = face.bbox.astype(int).tolist()
        x1, y1, x2, y2 = bbox
        # rescale bbox back to original image size if resized
        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        age = int(face.age) if hasattr(face, 'age') else None
        gender = "Male" if int(getattr(face, 'gender', 0)) == 1 else "Female"
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "age": age,
            "gender": gender
        })

    return jsonify({"detections": detections, "width": orig_w, "height": orig_h})


if __name__ == '__main__':
    # For local testing:
    app.run(host='0.0.0.0', port=5000, debug=False)
