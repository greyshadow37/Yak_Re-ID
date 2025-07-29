import gradio as gr
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from torchvision.models import mobilenet_v2
from torchvision import transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
import tempfile
import os

# --- Configuration ---
BASE_DIR = Path(__file__).parent
YOLO_MODEL_PATH = BASE_DIR / "weights" / "yolo11s_finetune2" / "weights" / "best.pt"
EXTRACTOR_MODEL_PATH = BASE_DIR / "weights" / "mobilenetv2_weights" / "best.pt"
TARGET_CLASS_IDS = list(range(10))  # classes to track

# --- Load YOLOv11 ---
def load_yolo_model():
    model = YOLO(YOLO_MODEL_PATH)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

# --- Load feature extractor ---
def load_feature_extractor():
    model = mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 512)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(EXTRACTOR_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# --- Tracker ---
def initialize_tracker():
    model = load_feature_extractor()
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        embedder_model_name=model,
        embedder_gpu=torch.cuda.is_available()
    )
    return tracker

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load once ---
yolo_model = load_yolo_model()
tracker = initialize_tracker()

# --- Image Processing ---
def process_image(input_image, confidence):
    image = input_image.convert("RGB")
    frame = np.array(image)

    # Detection
    results = yolo_model(image, conf=confidence, verbose=False)
    detections = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = float(result.conf[0])
        cls_id = int(result.cls[0])
        if cls_id in TARGET_CLASS_IDS and conf > confidence:
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, cls_id))

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Annotate
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return Image.fromarray(frame)

# --- Video Processing ---
def process_video(video_file, confidence):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close()
    video_path = tfile.name

    vid_cap = cv2.VideoCapture(video_path)
    frames = []
    while vid_cap.isOpened():
        ret, frame = vid_cap.read()
        if not ret:
            break

        # Detection
        results = yolo_model(frame, conf=confidence, verbose=False)
        detections = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = float(result.conf[0])
            cls_id = int(result.cls[0])
            if cls_id in TARGET_CLASS_IDS and conf > confidence:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls_id))

        # Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        # Annotate
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frames.append(frame)

    vid_cap.release()
    os.remove(video_path)

    # Save to temp output video
    height, width = frames[0].shape[:2]
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    for f in frames:
        writer.write(f)
    writer.release()

    return out_path

# --- Gradio Interface ---
def handle_input(image_or_video, confidence):
    if isinstance(image_or_video, Image.Image):
        return process_image(image_or_video, confidence)
    else:
        processed_video_path = process_video(image_or_video, confidence)
        return processed_video_path

with gr.Blocks(title="Yak Detection and Tracking") as demo:
    gr.Markdown("## 🦬 Yak Detection & Tracking with YOLOv11 + DeepSort")
    gr.Markdown("Upload an image or video to detect and track yaks. Adjust confidence threshold below.")
    
    with gr.Row():
        file_input = gr.File(label="Upload Image or Video", file_types=["image", "video"])
        confidence_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Confidence Threshold")

    output_image = gr.Image(label="Output Image", visible=False)
    output_video = gr.Video(label="Output Video", visible=False)

    def route_file(file, confidence):
        if file is None:
            return None, None
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext in [".jpg", ".jpeg", ".png"]:
            image = Image.open(file)
            result = process_image(image, confidence)
            return result, None
        elif file_ext == ".mp4":
            video_path = process_video(file, confidence)
            return None, video_path
        else:
            return None, None

    submit_btn = gr.Button("Run Detection")

    submit_btn.click(
        fn=route_file,
        inputs=[file_input, confidence_slider],
        outputs=[output_image, output_video]
    )

    gr.Markdown("Made for Yak monitoring and tracking research 🧠")

# --- Launch ---
if __name__ == "__main__":
    demo.launch(share=True)
