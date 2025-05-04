import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

class CustomClassifier:
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        """
        Initialize the custom classifier.
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a custom neural network model from scratch."""
        model = Sequential([
            # First hidden layer
            Dense(512, input_shape=(self.input_shape,)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            # Second hidden layer
            Dense(256),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            # Third hidden layer
            Dense(128),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, class_weights=None):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels (int indices)
            X_val: Validation features
            y_val: Validation labels (int indices)
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            History object with training metrics
        """
        print("Training shapes:", X_train.shape, y_train.shape)
        print("Validation shapes:", X_val.shape, y_val.shape)
        
        # Convert labels to one-hot encoding
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        # Compute class weights if not provided
        if class_weights is None and len(np.unique(y_train)) > 1:
            class_weights_array = compute_class_weight(
                'balanced', 
                classes=np.unique(y_train), 
                y=y_train
            )
            class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class indices
        """
        # Get probability distributions
        probabilities = self.model.predict(X)
        
        # Get class indices with highest probability
        return np.argmax(probabilities, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test labels (int indices)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Convert labels to one-hot encoding
        y_cat = to_categorical(y, self.num_classes)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X, y_cat, verbose=0)
        
        return loss, accuracy
    
    def plot_training_history(self, history):
        """
        Plot the training and validation metrics.
        
        Args:
            history: History object from model.fit()
        """
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()