'''Purpose: 
- Trains a YOLO model from pre-trained weights on a dataset and evaluates it.
Key Functionality:
- Trains the model and saves results.
- Evaluates and exports metrics to CSV.'''

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\Yak-Identification\repos\Yak_Re-ID\models\yolo11s.pt")
    
    model.train(
        data=r"D:\Yak-Identification\repos\Yak_Re-ID\data\data.yaml",
        epochs=30,
        imgsz=640,
        batch=-1,
        workers=2,  
        device=0,
        half=True,
        project='weights',
        name='yolo11s_results'
    )

    metrics = model.val()
    metrics.to_csv(r"D:\Yak-Identification\repos\Yak_Re-ID\weights\yolo11s_results\results.csv")
