'''Purpose: 
- Fine-tunes a pre-trained YOLO model on a custom dataset and evaluates it.
Key Functionality:
- Trains the model using a specified YAML dataset file.
- Saves evaluation metrics to a CSV file.
'''
from ultralytics import YOLO
import pandas as pd
import os

def main():
    # Define paths
    model_path = r"D:\Yak-Identification\repos\Yak_Re-ID\weights\yolo11s_results\weights\best.pt"
    data_yaml_path = r"D:\Yak-Identification\repos\Yak_Re-ID\data-finetune\1\data.yaml"
    output_dir = r"D:\Yak-Identification\repos\Yak_Re-ID\weights"
    run_name = 'yolo11s_finetune'
    results_csv_path = os.path.join(output_dir, run_name, 'results.csv')

    # Load the pretrained YOLO model
    model = YOLO(model_path)

    # Fine-tune (train further on your custom dataset)
    model.train(
        data=data_yaml_path,
        epochs=30,             
        imgsz=640,
        batch=-1,              
        workers=2,
        device=0,
        half=True,             
        project=output_dir,
        name=run_name
    )

    # Evaluate the model and save metrics
    metrics = model.val()
    
    # Convert metrics to DataFrame and save to CSV
    df = pd.DataFrame([metrics.results_dict])
    df.to_csv(results_csv_path, index=False)
    print(f"Evaluation results saved to: {results_csv_path}")

if __name__ == "__main__":
    main()
