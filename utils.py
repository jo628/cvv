import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_dataset(base_path, fold=1):
    """
    Load dataset from the specified path based on the train/test/val split files.
    
    Args:
        base_path: Path to the minc-2500 dataset
        fold: Which fold to use (1-5)
    
    Returns:
        Dictionary with train, test, and val data and labels
    """
    # Read categories
    with open(os.path.join(base_path, 'categories.txt'), 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    
    # Read train/test/val splits
    data = {'train': [], 'test': [], 'val': []}
    labels = {'train': [], 'test': [], 'val': []}
    
    for split in ['train', 'test', 'validate']:
        split_name = 'val' if split == 'validate' else split
        file_path = os.path.join(base_path, 'labels', f'{split}{fold}.txt')
        
        with open(file_path, 'r') as f:
            for line in f.readlines():
                img_path = line.strip()
                category = img_path.split('/')[1]
                full_path = os.path.join(base_path, img_path)
                
                data[split_name].append(full_path)
                labels[split_name].append(category_to_idx[category])
    
    return {
        'train_data': data['train'],
        'train_labels': labels['train'],
        'test_data': data['test'],
        'test_labels': labels['test'],
        'val_data': data['val'],
        'val_labels': labels['val'],
        'categories': categories,
        'category_to_idx': category_to_idx
    }

def visualize_results(images, predictions, true_labels, categories, num_samples=5):
    """
    Visualize prediction results.
    
    Args:
        images: List of images
        predictions: Predicted labels
        true_labels: True labels
        categories: List of category names
        num_samples: Number of samples to visualize
    """
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(f"Pred: {categories[predictions[i]]}\nTrue: {categories[true_labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_model(model, path="model_checkpoint.h5"):
    """Save model to disk"""
    model.save(path)
    print(f"Model saved to {path}")

def load_model(path="model_checkpoint.h5"):
    """Load model from disk"""
    from tensorflow.keras.models import load_model
    return load_model(path)