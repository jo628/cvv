import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Import custom modules
from utils import load_dataset, visualize_results, save_model
from preprocessing import ImagePreprocessor
from segmentation import CustomSegmenter
from feature_extraction import FeatureExtractor
from classification import CustomClassifier
from evaluation import ModelEvaluator

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("Starting Recycled Materials Classification Pipeline...")
    
    # 1. Load and prepare dataset
    print("\n[1] Loading dataset...")
    dataset_path = "/content/minc2500/minc-2500"
    dataset = load_dataset(dataset_path, fold=1)
    
    # Check if dataset is loaded correctly
    print(f"Categories: {dataset['categories']}")
    print(f"Number of training samples: {len(dataset['train_data'])}")
    print(f"Number of validation samples: {len(dataset['val_data'])}")
    print(f"Number of test samples: {len(dataset['test_data'])}")
    
    # 2. Preprocessing
    print("\n[2] Applying preprocessing...")
    preprocessor = ImagePreprocessor(target_size=(224, 224), apply_augmentation=True)
    
    # Process training data
    X_train_preprocessed = preprocessor.batch_preprocess(dataset['train_data'], segment=False)
    
    # Process validation data (no augmentation for validation)
    preprocessor.apply_augmentation = False
    X_val_preprocessed = preprocessor.batch_preprocess(dataset['val_data'], segment=False)
    
    # Process test data (no augmentation for test)
    X_test_preprocessed = preprocessor.batch_preprocess(dataset['test_data'], segment=False)
    
    # 3. Segmentation
    print("\n[3] Applying segmentation...")
    segmenter = CustomSegmenter(method='clustering', n_clusters=5)
    
    # Only segment a subset for visualization (segmentation is optional in this pipeline)
    sample_indices = np.random.choice(range(len(X_train_preprocessed)), 3, replace=False)
    sample_images = [X_train_preprocessed[i] for i in sample_indices]
    
    # Show sample segmentations
    for i, img in enumerate(sample_images):
        segmented = segmenter.segment(img)
        segmenter.visualize_segmentation(img, segmented)
    
    # 4. Feature Extraction
    print("\n[4] Extracting features...")
    feature_extractor = FeatureExtractor(input_shape=(224, 224, 3))
    
    # Extract features from preprocessed images
    X_train_features = feature_extractor.batch_extract_features(X_train_preprocessed)
    X_val_features = feature_extractor.batch_extract_features(X_val_preprocessed)
    X_test_features = feature_extractor.batch_extract_features(X_test_preprocessed)
    
    print(f"Training features shape: {X_train_features.shape}")
    print(f"Validation features shape: {X_val_features.shape}")
    print(f"Test features shape: {X_test_features.shape}")
    
    # 5. Classification
    print("\n[5] Training classifier...")
    classifier = CustomClassifier(
        input_shape=X_train_features.shape[1],
        num_classes=len(dataset['categories']),
        learning_rate=0.001
    )
    
    # Train the classifier
    history = classifier.train(
        X_train=X_train_features,
        y_train=dataset['train_labels'],
        X_val=X_val_features,
        y_val=dataset['val_labels'],
        batch_size=32,
        epochs=100
    )
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # 6. Evaluation
    print("\n[6] Evaluating model...")
    evaluator = ModelEvaluator(dataset['categories'])
    
    # Make predictions on test set
    y_pred = classifier.predict(X_test_features)
    
    # Calculate and print metrics
    evaluator.print_metrics(dataset['test_labels'], y_pred, dataset_name="Test")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(dataset['test_labels'], y_pred)
    
    # Display some examples
    # First, load original images for better visualization
    test_images = []
    for path in tqdm(dataset['test_data'][:20], desc="Loading test images"):
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_images.append(img)
    
    # Show examples of correct and incorrect predictions
    evaluator.plot_examples(
        test_images,
        dataset['test_labels'][:20],
        y_pred[:20],
        num_examples=5
    )
    
    # 7. Save the model
    save_model(classifier.model, "recycled_materials_classifier.h5")
    
    print("\nRecycled Materials Classification Pipeline completed successfully!")

if __name__ == "__main__":
    main()