'''Purpose: 
- Stitches a sequence of images from a folder into a video, typically used to visualize inference results.
Key Functionality:
- Reads .jpg or .png images from a specified folder.
- Sorts images numerically based on filenames.
- Creates a video using OpenCV at a specified frame rate (default: 10 fps).
'''
import cv2
import os

def stitch_frames_to_video(folder, output_path, fps=10):
    images = [img for img in os.listdir(folder) if img.endswith((".jpg", ".png"))]
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  

    if not images:
        raise ValueError("No images found to stitch into video.")

    first_image_path = os.path.join(folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    for img_name in images:
        img_path = os.path.join(folder, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")

predict_folder = r"D:\Yak-Identification\repos\Yak_Re-ID\inference\inference_results_f3"
output_video = os.path.join(predict_folder, "yolo_output_video.mp4")

stitch_frames_to_video(predict_folder, output_video, fps=10)
