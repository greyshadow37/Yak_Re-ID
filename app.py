from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
import torch
import torch.nn as nn
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models import mobilenet_v2
from torchvision import transforms
from pathlib import Path
import io
import time
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = Path(__file__).parent
YOLO_MODEL_PATH = BASE_DIR / "weights" / "yolo11s_finetune2" / "weights" / "best.pt"
EXTRACTOR_MODEL_PATH = BASE_DIR / "weights" / "mobilenetv2_weights" / "best.pt"
TARGET_CLASS_IDS = list(range(10))
CONFIDENCE_THRESHOLD = 0.5

# --- Model Loading ---
def load_yolo_model():
    """Loads the custom YOLOv11 model."""
    try:
        model = YOLO(YOLO_MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logger.info(f"YOLO model loaded on {device}")
        return model, device
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        return None, str(e)

def load_feature_extractor():
    """Loads the MobileNetV2 feature extractor."""
    try:
        model = mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 512)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(EXTRACTOR_MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        logger.info("Feature extractor loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading feature extractor: {str(e)}")
        return None, str(e)

def initialize_tracker():
    """Initializes the DeepSort tracker with the feature extractor."""
    reid_model = load_feature_extractor()
    if reid_model is None:
        logger.error("Failed to initialize tracker: Feature extractor not loaded")
        return None, "Failed to load feature extractor"
    try:
        tracker = DeepSort(
            max_age=30,
            n_init=3,
            embedder_model_name=reid_model,
            embedder_gpu=torch.cuda.is_available()
        )
        logger.info("DeepSort tracker initialized")
        return tracker
    except Exception as e:
        logger.error(f"Error initializing DeepSort tracker: {str(e)}")
        return None, str(e)

# --- Image Transformation for Feature Extractor ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load models and tracker
yolo_model, device = load_yolo_model()
tracker = initialize_tracker()

# --- Safe File Deletion with Retry ---
def safe_remove(path, retries=5, delay=1):
    """Attempts to delete a file with retries to handle locking issues."""
    if not path or not os.path.exists(path):
        return
    for attempt in range(retries):
        try:
            os.remove(path)
            logger.info(f"Successfully deleted file: {path}")
            return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to delete {path}: {str(e)}")
            time.sleep(delay)
            gc.collect()
    logger.error(f"Failed to delete {path} after {retries} attempts")

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health():
    if yolo_model is None or tracker is None:
        return jsonify({"status": "error", "message": "Models failed to load"}), 500
    return jsonify({"status": "healthy", "device": device}), 200

# --- Image Processing Endpoint ---
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.jpg', '.jpeg', '.png']:
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        # Load and process image
        image = Image.open(file).convert("RGB")
        frame = np.array(image)

        # Run YOLO detection
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = float(result.conf[0])
            cls_id = int(result.cls[0])
            if cls_id in TARGET_CLASS_IDS and conf > CONFIDENCE_THRESHOLD:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls_id))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw results
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save processed image to temporary file
        _, buffer = cv2.imencode('.png', frame)
        processed_image = io.BytesIO(buffer)
        return send_file(processed_image, mimetype='image/png', download_name='processed_image.png')

    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

# --- Video Processing Endpoint ---
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if os.path.splitext(file.filename)[1].lower() != '.mp4':
        return jsonify({"error": "Unsupported file type"}), 400

    video_path = None
    output_path = None
    vid_cap = None
    out = None

    try:
        # Save uploaded video to temporary file
        video_path = os.path.join(tempfile.gettempdir(), f"input_{int(time.time() * 1000)}.mp4")
        logger.info(f"Saving uploaded video to {video_path}")
        with open(video_path, 'wb') as f:
            f.write(file.read())
            f.flush()
            os.fsync(f.fileno())  # Ensure file is fully written

        # Initialize video capture
        vid_cap = cv2.VideoCapture(video_path)
        if not vid_cap.isOpened():
            logger.error("Failed to open video file")
            return jsonify({"error": "Failed to open video file"}), 500

        # Get video properties
        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties: FPS={fps}, Width={width}, Height={height}, Frames={frame_count}")

        # Initialize output video
        output_path = os.path.join(tempfile.gettempdir(), f"output_{int(time.time() * 1000)}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H264 codec for compatibility
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.error("Failed to initialize video writer")
            return jsonify({"error": "Failed to initialize video writer"}), 500

        # Process video frames
        processed_frames = 0
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if not ret:
                logger.info(f"Reached end of video after {processed_frames} frames")
                break
            if frame is None or frame.size == 0:
                logger.warning("Empty frame encountered")
                continue

            # Run YOLO detection
            results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            detections = []
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                conf = float(result.conf[0])
                cls_id = int(result.cls[0])
                if cls_id in TARGET_CLASS_IDS and conf > CONFIDENCE_THRESHOLD:
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, cls_id))

            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Draw results
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write frame to output
            out.write(frame)
            processed_frames += 1
            logger.debug(f"Processed frame {processed_frames}")

        if processed_frames == 0:
            logger.error("No frames processed")
            return jsonify({"error": "No frames processed in the video"}), 500

        logger.info(f"Video processing completed: {processed_frames} frames written")

        # Verify output file
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("Output video file is missing or empty")
            return jsonify({"error": "Output video file is missing or empty"}), 500

        # Send processed video with download disposition
        response = send_file(
            output_path,
            mimetype='video/mp4',
            download_name='processed_video.mp4',
            as_attachment=True  # Force download
        )

    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

    finally:
        # Ensure resources are released
        if vid_cap is not None:
            vid_cap.release()
            logger.info("Video capture released")
        if out is not None:
            out.release()
            logger.info("Video writer released")
        # Force garbage collection
        gc.collect()
        # Clean up temporary files
        for path in [video_path, output_path]:
            safe_remove(path)

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)