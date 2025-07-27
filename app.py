'''Purpose:
- This application detects and tracks yaks in images and videos using advanced computer vision techniques for accurate identification and re-identification.
Key Functionality:
- Upload and Process: Upload images or videos for analysis.
- Yak Detection: Detect yaks using a custom YOLOv11 model with adjustable confidence threshold.
- Tracking: Track yaks across video frames with DeepSort for consistent identities.
- Interactive UI: View real-time detection and tracking results.'''
import streamlit as st
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

# --- Page Configuration ---
st.set_page_config(
    page_title="Yak Tracking and Re-Identification",
    page_icon="🦬",
    layout="wide"
)

# --- Configuration ---
YOLO_MODEL_PATH = r"D:\Yak-Identification\repos\Yak_Re-ID\weights\yolo11s_finetune2\weights\best.pt"
EXTRACTOR_MODEL_PATH = r"D:\Yak-Identification\repos\Yak_Re-ID\weights\mobilenetv2_weights\best.pt"
TARGET_CLASS_IDS = list(range(10))  
CONFIDENCE_THRESHOLD = 0.5

# --- Model Loading ---
@st.cache_resource
def load_yolo_model():
    """Loads the custom YOLOv11 model."""
    try:
        model = YOLO(YOLO_MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        st.success(f"✅ YOLO model loaded successfully on {device}.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

@st.cache_resource
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
        st.success("✅ Feature extractor loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading feature extractor: {e}")
        return None

# --- Initialize Tracker ---
@st.cache_resource
def initialize_tracker():
    """Initializes the DeepSort tracker with the feature extractor."""
    reid_model = load_feature_extractor()
    if reid_model is None:
        return None
    try:
        tracker = DeepSort(
            max_age=30,
            n_init=3,
            embedder_model_name=reid_model,
            embedder_gpu=torch.cuda.is_available()
        )
        st.success("✅ DeepSort tracker initialized.")
        return tracker
    except Exception as e:
        st.error(f"Error initializing DeepSort tracker: {e}")
        return None

# --- Image Transformation for Feature Extractor ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Main App UI ---
st.title("🦬 Yak Tracking and Re-Identification")
st.write("Upload an image or video to detect and track yaks using a custom YOLOv11 model and DeepSort.")
st.info("This model is trained specifically for yak detection and re-identification. Adjust the confidence threshold to filter detections.")

# Load models and tracker
yolo_model = load_yolo_model()
tracker = initialize_tracker()

# Confidence threshold slider
confidence = st.slider("Select Model Confidence", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.01)

if yolo_model and tracker:
    uploaded_file = st.file_uploader(
        "Choose an image or video...",
        type=["jpg", "jpeg", "png", "mp4"]
    )

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # --- IMAGE PROCESSING ---
        if file_extension in [".jpg", ".jpeg", ".png"]:
            image = Image.open(uploaded_file).convert("RGB")
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)

            st.subheader("Detection Results")
            with st.spinner('Detecting objects...'):
                # Run YOLO detection
                results = yolo_model(image, conf=confidence, verbose=False)
                detections = []
                for result in results[0].boxes:
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    conf = float(result.conf[0])
                    cls_id = int(result.cls[0])
                    if cls_id in TARGET_CLASS_IDS and conf > confidence:
                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], conf, cls_id))

                # Update tracker
                frame = np.array(image)
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

                st.image(frame, caption="Processed Image with Tracked Yaks", use_column_width=True)

        # --- VIDEO PROCESSING ---
        elif file_extension == ".mp4":
            st.subheader("Uploaded Video")
            video_bytes = uploaded_file.getvalue()
            st.video(video_bytes)

            st.subheader("Tracking Results")
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_bytes)
            video_path = tfile.name
            tfile.close()

            vid_cap = None
            try:
                vid_cap = cv2.VideoCapture(video_path)
                with st.spinner('Processing video... This may take a while.'):
                    frame_placeholder = st.empty()
                    while vid_cap.isOpened():
                        ret, frame = vid_cap.read()
                        if not ret:
                            break

                        # Run YOLO detection
                        results = yolo_model(frame, conf=confidence, verbose=False)
                        detections = []
                        for result in results[0].boxes:
                            x1, y1, x2, y2 = map(int, result.xyxy[0])
                            conf = float(result.conf[0])
                            cls_id = int(result.cls[0])
                            if cls_id in TARGET_CLASS_IDS and conf > confidence:
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

                        # Display frame
                        frame_placeholder.image(frame, channels="BGR", caption="Processing...")

                st.success("Video processing complete.")

            finally:
                if vid_cap is not None:
                    vid_cap.release()
                if os.path.exists(video_path):
                    os.remove(video_path)
else:
    st.error("Failed to load models or tracker. Please check the model paths and try again.")