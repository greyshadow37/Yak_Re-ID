
# Yak Re-Identification System

An end-to-end computer vision pipeline designed to detect, track, and analyze yak behavior using deep learning and object tracking techniques.

## 🚀 Project Overview

This system performs:
- **Detection**: Uses a fine-tuned YOLO model to detect yaks in video frames.
- **Re-Identification & Tracking**: Leverages [`deep-sort-realtime`](https://pypi.org/project/deep-sort-realtime/) to assign consistent IDs across frames.
- **Behavior Analysis**: Analyzes movement patterns to infer behaviors such as shed occupancy.
- **Visualization**: Outputs annotated frames and compiles videos for further analysis.

## 🗂 Repository Structure

| Script                      | Purpose                                                                        |
| --------------------------- | ------------------------------------------------------------------------------ |
| `clear_cuda.py`             | Frees GPU memory using `torch.cuda.empty_cache()`.                             |
| `split_yolo_data.py`        | Splits dataset into training, validation, and test sets for YOLO.              |
| `fine_tune_yolo.py`         | Fine-tunes a YOLO model on custom data and logs metrics.                       |
| `yolo-detector.py`          | Trains and evaluates a YOLO model from pre-trained weights.                    |
| `inference.py`              | Runs YOLO detection with Deep SORT for re-identification/tracking on videos.   |
| `video_generation.py`       | Combines annotated frames into a video for visualization.                      |
| `fine-tune-weights.py`      | Legacy script for MobileNetV2-based re-ID (may be deprecated with Deep SORT).  |
| `tuned_model_test.py`       | Legacy script for testing re-ID models.                                        |

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy, Pandas, Matplotlib, Scikit-learn
- Pillow (PIL)
- [`deep-sort-realtime`](https://pypi.org/project/deep-sort-realtime/)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Yak_Re-ID.git
    cd Yak_Re-ID
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Organize your dataset as described in the Dataset Structure section and update script paths as needed.

## 🗃️ Dataset Structure

### For YOLO Training (under `data-finetune/`):

```
data-finetune/
└── 2/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── train.txt
    └── data.yaml
```

### For Deep SORT or Cropped Test Data (optional, under `data-cropped/`):

```
data-cropped/
└── test/
    ├── yak_001/
    ├── yak_002/
    └── ...
```

### 🧪 Usage

### 1. Data Preparation

Split your dataset into training, validation, and test sets:
```bash
python split_yolo_data.py
```

### 2. Model Training

Train the YOLO model using one of the following scripts:
```bash
python yolo-detector.py
```
or fine-tune an existing model:
```bash
python fine_tune_yolo.py
```
*Update paths for your dataset and weights inside the scripts as needed.*

### 3. Inference and Tracking

Run the inference script to detect yaks and track them across frames:
```bash
python inference.py
```
This step applies Deep SORT for re-identification and saves annotated frames in a designated output folder (e.g., `inference/inference_results_f3/`).

### 4. Video Generation

Combine the annotated frames into a final video:
```bash
python video_generation.py
```

### 5. GPU Memory Management

If you encounter CUDA memory issues during training or inference, free up GPU memory with:
```bash
python clear_cuda.py
```

## 📊 Results & Output

- **YOLO Training Output**:
  - Location: `weights/yolo11s_results/`
  - Contains: Model weights, training logs, and `results.csv`.

- **Inference Output**:
  - Annotated frames saved in a directory such as `inference/inference_results/`.

- **Final Video**:
  - Generated video saved as `yolo_output_video.mp4`.

## 📝 Notes

- Adjust hyperparameters (e.g., image size, epochs, batch size) in the training scripts according to your hardware.
- Ensure input videos are compatible with OpenCV (e.g., have a `.mp4` extension).
- Update file paths and configuration settings in the scripts to suit your environment.
