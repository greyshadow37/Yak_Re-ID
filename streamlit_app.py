import streamlit as st
import requests
from PIL import Image
import os
import tempfile
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Yak Tracking and Re-Identification",
    page_icon="🦬",
    layout="wide"
)

# --- Configuration ---
API_URL = "http://localhost:5000"  # Update with Render URL after deployment
CONFIDENCE_THRESHOLD = 0.5

# --- Main App UI ---
st.title("🦬 Yak Tracking and Re-Identification")
st.write("Upload an image or video to detect and track yaks using a custom YOLOv11 model and DeepSort.")
st.info("This model is trained specifically for yak detection and re-identification. Adjust the confidence threshold to filter detections.")

# Check API health
try:
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        st.success(f"✅ API is healthy, running on {response.json()['device']}.")
    else:
        st.error("API is not responding. Please check the backend service.")
except:
    st.error("Failed to connect to the API. Ensure the backend is running.")

# Confidence threshold slider
confidence = st.slider("Select Model Confidence", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.01)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image or video...",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    # --- IMAGE PROCESSING ---
    if file_extension in [".jpg", ".jpeg", ".png"]:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        st.subheader("Detection Results")
        with st.spinner('Detecting objects...'):
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                image.save(tmp.name)
                files = {'file': open(tmp.name, 'rb')}
                response = requests.post(
                    f"{API_URL}/process_image",
                    files=files,
                    params={'confidence': confidence}
                )
                files['file'].close()
                os.remove(tmp.name)

            if response.status_code == 200:
                processed_image = Image.open(io.BytesIO(response.content))
                st.image(processed_image, caption="Processed Image with Tracked Yaks", use_column_width=True)
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

    # --- VIDEO PROCESSING ---
    elif file_extension == ".mp4":
        st.subheader("Uploaded Video")
        video_bytes = uploaded_file.getvalue()
        st.video(video_bytes)

        st.subheader("Tracking Results")
        with st.spinner('Processing video... This may take a while.'):
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                tmp.close()
                files = {'file': open(tmp.name, 'rb')}
                response = requests.post(
                    f"{API_URL}/process_video",
                    files=files,
                    params={'confidence': confidence}
                )
                files['file'].close()
                os.remove(tmp.name)

            if response.status_code == 200:
                # Save processed video to temporary file for display and download
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                    tmp_out.write(response.content)
                    tmp_out.close()
                    # Provide download button
                    with open(tmp_out.name, "rb") as f:
                        st.download_button(
                            label="Download Processed Video",
                            data=f,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
                    # Attempt to display video (optional)
                    try:
                        st.video(tmp_out.name)
                        st.success("Video processing complete. Download the video to ensure playback.")
                    except Exception as e:
                        st.warning(f"Video preview may not work in this browser: {str(e)}. Please download the video to play it.")
                    os.remove(tmp_out.name)
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

else:
    st.info("Please upload an image or video to start processing.")