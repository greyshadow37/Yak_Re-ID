<<<<<<< HEAD
'''Purpose: 
- Evaluates a fine-tuned MobileNetV2 model on a test dataset, computing performance metrics.
Key Functionality:
- Loads test images and computes embeddings.
- Calculates similarity matrices and metrics (precision, recall, F1, ROC AUC).
- Saves results and plots to a directory.
'''

=======
>>>>>>> e5614c173 (final changes)
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
class YakDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Load test data
def load_test_dataset(test_dir, label_map):
    image_paths, labels = [], []
    for yak_id in sorted(os.listdir(test_dir)):
        yak_path = os.path.join(test_dir, yak_id)
        if os.path.isdir(yak_path) and yak_id in label_map:
            label_idx = label_map[yak_id]
            for img_name in os.listdir(yak_path):
                if img_name.lower().endswith(('.jpg', '.png')):
                    image_paths.append(os.path.join(yak_path, img_name))
                    labels.append(label_idx)
    return image_paths, labels

# Compute cosine similarity matrix
def compute_similarity_matrix(embeddings):
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(normed, normed.T)

# Metrics computation
def compute_metrics(similarity, labels, thresholds=np.linspace(0, 1, 100)):
    N = len(labels)
    pairwise_true = []
    similarity_scores = []

    for i in range(N):
        for j in range(i + 1, N):
            pairwise_true.append(int(labels[i] == labels[j]))
            similarity_scores.append(similarity[i, j])

    pairwise_true = np.array(pairwise_true)
    similarity_scores = np.array(similarity_scores)

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pred = similarity_scores > t
        p, r, f, _ = precision_recall_fscore_support(pairwise_true, pred, average='binary', zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    best_f1 = f1s[best_idx]

    fpr, tpr, _ = roc_curve(pairwise_true, similarity_scores)
    roc_auc = auc(fpr, tpr)

    return {
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1s,
        'best_threshold': best_threshold
    }

# Plot and save metrics
def save_metrics(metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1', 'AUC'],
        'Value': [metrics['precision'], metrics['recall'], metrics['f1'], metrics['roc_auc']]
    }).to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

    plt.figure()
    plt.plot(metrics['fpr'], metrics['tpr'], label=f'AUC = {metrics["roc_auc"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(metrics['recalls'], metrics['precisions'])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(metrics['thresholds'], metrics['f1_scores'])
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 vs. Threshold")
    plt.savefig(os.path.join(results_dir, 'f1_vs_threshold.png'))
    plt.close()

# Main test loop
def main():
    test_dir = r'D:\Yak-Identification\repos\Yak_Re-ID\data-cropped\test'
    weights_path = r'D:\Yak-Identification\repos\Yak_Re-ID\weights\mobilenetv2_weights\best.pt'
    results_dir = r'D:\Yak-Identification\repos\Yak_Re-ID\weights\mobilenetv2_test_results'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_map = {}
    current_label = 0
    for yak_id in sorted(os.listdir(test_dir)):
        yak_path = os.path.join(test_dir, yak_id)
        if os.path.isdir(yak_path):
            label_map[yak_id] = current_label
            current_label += 1

    image_paths, labels = load_test_dataset(test_dir, label_map)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = YakDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = models.mobilenet_v2(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model.last_channel, 512)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("✅ Model loaded successfully")

    model = model.to(device)
    print("✅ Model moved to device")

    model.eval()

    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            features = model(images)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(lbls.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    similarity = compute_similarity_matrix(embeddings)
    metrics = compute_metrics(similarity, labels)
    save_metrics(metrics, results_dir)

    print("\n=== Embedding Evaluation ===")
    print(f"Best Threshold: {metrics['best_threshold']:.3f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1 Score:       {metrics['f1']:.4f}")
    print(f"ROC AUC:        {metrics['roc_auc']:.4f}")

if __name__ == '__main__':
    main()
