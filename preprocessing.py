import cv2
import numpy as np
from skimage import exposure
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), apply_augmentation=True):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            apply_augmentation: Whether to apply augmentation
        """
        self.target_size = target_size
        self.apply_augmentation = apply_augmentation
        
        # Create ImageDataGenerator for augmentation
        self.augmenter = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        ) if apply_augmentation else None
    
    def resize_image(self, image):
        """Resize image to target size"""
        return cv2.resize(image, self.target_size)
    
    def normalize_image(self, image):
        """Normalize pixel values to [0,1]"""
        return image.astype(np.float32) / 255.0
    
    def reduce_noise(self, image):
        """Apply noise reduction"""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def adjust_contrast(self, image):
        """Adjust contrast using histogram equalization"""
        # Convert to YUV color space
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Apply histogram equalization to Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # Convert back to BGR
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    def convert_color_space(self, image, target_space='hsv'):
        """Convert image color space"""
        if target_space.lower() == 'hsv':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif target_space.lower() == 'gray':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif target_space.lower() == 'lab':
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return image
    
    def apply_thresholding(self, image, method='otsu'):
        """Apply thresholding to image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        if method == 'otsu':
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        else:
            _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        return thresholded
    
    def apply_blur(self, image, method='gaussian', kernel_size=5):
        """Apply blurring to image"""
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == 'median':
            return cv2.medianBlur(image, kernel_size)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        return image
    
    def apply_sharpen(self, image):
        """Apply sharpening to image"""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def apply_morphology(self, image, operation='opening', kernel_size=5):
        """Apply morphological operations"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'gradient':
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        elif operation == 'dilation':
            return cv2.dilate(image, kernel, iterations=1)
        elif operation == 'erosion':
            return cv2.erode(image, kernel, iterations=1)
        return image
    
    def augment_image(self, image):
        """Apply data augmentation"""
        if not self.apply_augmentation:
            return image
            
        # Convert to format expected by ImageDataGenerator
        image = np.expand_dims(image, axis=0)
        
        # Get a random augmentation
        aug_iter = self.augmenter.flow(image, batch_size=1)
        aug_image = next(aug_iter)[0].astype(np.uint8)
        
        return aug_image
    
    def preprocess(self, image_path, segment=False):
        """
        Apply the full preprocessing pipeline to an image.
        
        Args:
            image_path: Path to image file
            segment: Whether the output will be used for segmentation
            
        Returns:
            Preprocessed image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Apply preprocessing steps
        image = self.resize_image(image)
        image = self.reduce_noise(image)
        image = self.adjust_contrast(image)
        
        # For segmentation, keep the original color space and don't normalize
        if not segment:
            # Random choice of enhancements for data diversity
            enhancement_choices = ['sharpen', 'blur', 'none']
            choice = random.choice(enhancement_choices)
            
            if choice == 'sharpen':
                image = self.apply_sharpen(image)
            elif choice == 'blur':
                image = self.apply_blur(image, method=random.choice(['gaussian', 'median']), 
                                      kernel_size=random.choice([3, 5]))
            
            # Apply augmentation with probability (only for training)
            if self.apply_augmentation and random.random() < 0.5:
                image = self.augment_image(image)
            
            # Always normalize at the end for deep learning models
            image = self.normalize_image(image)
            
        return image
    
    def batch_preprocess(self, image_paths, segment=False, with_tqdm=True):
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image paths
            segment: Whether the output will be used for segmentation
            with_tqdm: Whether to use tqdm for progress tracking
            
        Returns:
            Preprocessed images
        """
        if with_tqdm:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="Preprocessing images")
        else:
            iterator = image_paths
            
        return [self.preprocess(img_path, segment) for img_path in iterator]