'''Purpose: 
- Similar to fine-tune-weights.py, evaluates a MobileNetV2 model on a test dataset.
Key Functionality:
- Computes embeddings, similarity matrices, and metrics.
- Includes confusion matrix visualization.'''
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Dataset class 
class YakDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

# Load test dataset
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

# Cosine similarity matrix
def compute_similarity_matrix(embeddings):
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(normed, normed.T)

# Compute evaluation metrics
def compute_metrics(model, dataset, device):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    model.eval()

    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        print(f"Loaded {len(loader)} batches")
        for i, batch in enumerate(loader):
            print(f"Processing batch {i + 1}/{len(loader)}")
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = compute_similarity_matrix(embeddings)

    pairwise_true = (true_labels[:, None] == true_labels[None, :]).flatten()
    pairwise_pred = (similarity.flatten() > 0.5)
    precision, recall, f1, _ = precision_recall_fscore_support(pairwise_true, pairwise_pred, average='binary')

    fpr, tpr, _ = roc_curve(pairwise_true, similarity.flatten())
    roc_auc = auc(fpr, tpr)

    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1_scores = [], [], []
    for t in thresholds:
        pred = similarity.flatten() > t
        p, r, f, _ = precision_recall_fscore_support(pairwise_true, pred, average='binary', zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f)

    predicted_ids = np.argmax(similarity, axis=1)
    cm = confusion_matrix(true_labels, true_labels[predicted_ids])
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'cm': cm,
        'cm_normalized': cm_normalized
    }

# Save plots and metrics to disk
def save_metrics(metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1', 'AUC'],
        'Value': [metrics['precision'], metrics['recall'], metrics['f1'], metrics['roc_auc']]
    }).to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

    pd.DataFrame(metrics['cm']).to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))
    pd.DataFrame(metrics['cm_normalized']).to_csv(os.path.join(results_dir, 'confusion_matrix_normalized.csv'))

    plt.plot(metrics['fpr'], metrics['tpr'], label=f'AUC = {metrics["roc_auc"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.clf()

    plt.plot(metrics['recalls'], metrics['precisions'])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'))
    plt.clf()

    plt.plot(metrics['thresholds'], metrics['f1_scores'])
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 vs. Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'f1_vs_threshold.png'))
    plt.clf()

    plt.imshow(metrics['cm_normalized'], cmap='Blues')
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

# Entry point
def main():
    test_dir = r'D:\Yak-Identification\repos\Yak_Re-ID\data-cropped\test'
    weights_path = r'D:\Yak-Identification\repos\Yak_Re-ID\weights\mobilenetv2_weights\best.pt'
    results_dir = r'D:\Yak-Identification\repos\Yak_Re-ID\weights\mobilenetv2_test_results'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build label map from test directory
    label_map = {}
    current_label = 0
    for yak_id in sorted(os.listdir(test_dir)):
        yak_path = os.path.join(test_dir, yak_id)
        if os.path.isdir(yak_path):
            if yak_id not in label_map:
                label_map[yak_id] = current_label
                current_label += 1

    # Load dataset
    image_paths, labels = load_test_dataset(test_dir, label_map)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = YakDataset(image_paths, labels, transform)

    # Load model
    model = models.mobilenet_v2(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model.last_channel, 512)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)

    # Evaluate
    print("Evaluating model on test set...")
    metrics = compute_metrics(model, dataset, device)
    save_metrics(metrics, results_dir)

    # Print summary
    print(f"\nPrecision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1 Score:  {metrics['f1']:.2f}")
    print(f"AUC:       {metrics['roc_auc']:.2f}")

if __name__ == '__main__':
    main()
