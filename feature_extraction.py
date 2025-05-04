import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import numpy as np
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, input_shape=(224, 224, 3), include_top=False):
        """
        Initialize the feature extractor using MobileNetV2.
        
        Args:
            input_shape: Input shape for the model (height, width, channels)
            include_top: Whether to include the fully-connected layer at the top of the network
        """
        self.input_shape = input_shape
        
        # Load MobileNetV2 model pre-trained on ImageNet
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=include_top,
            weights='imagenet'
        )
        
        # Use the output of the last layer as features
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.layers[-1].output
        )
        
        # Freeze the base model to use it as a feature extractor
        self.model.trainable = False
        
        print(f"Feature extractor initialized with output shape: {self.model.output_shape}")
    
    def extract_features(self, image):
        """
        Extract features from a single image.
        
        Args:
            image: Input image (normalized to [0,1])
            
        Returns:
            Extracted features
        """
        # Ensure the image has the right shape with batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Ensure float32 dtype
        image = image.astype(np.float32)
        
        # Extract features
        features = self.model.predict(image)
        
        return features[0]  # Remove batch dimension
    
    def batch_extract_features(self, images, batch_size=32):
        """
        Extract features from a batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for prediction
            
        Returns:
            Extracted features for all images
        """
        # Ensure all images are float32 and have the right shape
        processed_images = []
        for img in images:
            # Skip None or invalid images
            if img is None or img.size == 0:
                continue
                
            if len(img.shape) == 2:  # Grayscale to RGB
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 1:  # Single channel to RGB
                img = np.concatenate([img, img, img], axis=-1)
            elif img.shape[-1] > 3:  # More than 3 channels
                img = img[:, :, :3]
                
            # Ensure image has the correct shape
            if img.shape[:2] != self.input_shape[:2]:
                img = tf.image.resize(img, self.input_shape[:2]).numpy()
                
            processed_images.append(img)
        
        if not processed_images:
            return np.array([])
        
        # Convert to numpy array
        processed_images = np.array(processed_images, dtype=np.float32)
        
        # Extract features in batches
        n_samples = len(processed_images)
        features = []
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting features"):
            batch = processed_images[i:i+batch_size]
            batch_features = self.model.predict(batch, verbose=0)
            features.append(batch_features)
        
        # Concatenate all features
        return np.vstack(features)