'''Purpose: 
- Runs inference on a video using a pre-trained YOLO model and saves frames with detections as images.
Key Functionality:
- Loads a YOLO model and processes a video file.
- Saves each frame with detections in a specified output directory.'''

from ultralytics import YOLO
import os

if __name__ == "__main__":
    model = YOLO(r"D:\Yak-Identification\repos\Yak_Re-ID\weights\yolo11s_results\weights\best.pt")

    source = r"D:\Yak-Identification\repos\Yak_Re-ID\inference\test3.mp4"

    output_dir = r"D:\Yak-Identification\repos\Yak_Re-ID\inference\inference_results_f3"
    os.makedirs(output_dir, exist_ok=True)

    results = model(source, imgsz=640, conf=0.25)

    for i, result in enumerate(results):
        img = result.plot()  
        save_path = os.path.join(output_dir, f"result_{i:03d}.jpg")
        result.save(filename=save_path)
        print(f"Saved: {save_path}")
