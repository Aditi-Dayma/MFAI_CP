"""
cnn_model.py
CNN-based face recognition using TensorFlow / Keras.
"""
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.visualization import plot_accuracy_loss, array_to_base64


class CNNFaceRecognizer:
    """CNN-based face recognition using a custom convolutional architecture."""

    def __init__(self):
        self.model = None
        self.label_names = None
        self.metrics = {}
        self.history = None
        self.img_size = (100, 100)

    def _build_model(self, n_classes):
        """Build a custom CNN architecture."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.img_size[0], self.img_size[1], 1)),

            # Block 1
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classifier
            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, images, labels, label_names, epochs=20):
        """
        Train the CNN model on the given face images.

        Args:
            images: list of 2D numpy arrays (grayscale faces, 100x100)
            labels: list of string labels
            label_names: sorted unique label names
            epochs: number of training epochs
        Returns:
            dict with training results and metrics
        """
        self.label_names = label_names
        n_classes = len(label_names)

        start_time = time.time()

        # Prepare data
        X = np.array([img.astype(np.float32) / 255.0 for img in images])
        X = X.reshape(-1, self.img_size[0], self.img_size[1], 1)
        y = np.array([label_names.index(l) for l in labels])

        # Train/test split
        if len(X) < 4:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
            )

        # Build model
        self.model = self._build_model(n_classes)

        # Data augmentation for small datasets
        import tensorflow as tf
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=min(32, len(X_train)),
            verbose=0
        )

        training_time = time.time() - start_time

        # Store history
        self.history = {
            'accuracy': [float(v) for v in history.history['accuracy']],
            'loss': [float(v) for v in history.history['loss']],
            'val_accuracy': [float(v) for v in history.history.get('val_accuracy', [])],
            'val_loss': [float(v) for v in history.history.get('val_loss', [])],
        }

        # Evaluate
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        accuracy = accuracy_score(y_test, y_pred)

        # Generate charts
        accuracy_loss_chart = plot_accuracy_loss(self.history)

        self.metrics = {
            'accuracy': float(accuracy),
            'training_time': round(training_time, 4),
            'prediction_speed': 0,
            'epochs': epochs,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'final_train_accuracy': round(self.history['accuracy'][-1], 4),
            'final_val_accuracy': round(self.history['val_accuracy'][-1], 4) if self.history['val_accuracy'] else None,
            'final_train_loss': round(self.history['loss'][-1], 4),
            'final_val_loss': round(self.history['val_loss'][-1], 4) if self.history['val_loss'] else None,
        }

        return {
            'metrics': self.metrics,
            'accuracy_loss_chart': accuracy_loss_chart,
            'history': self.history,
        }

    def predict(self, test_image):
        """
        Predict the identity of a test face image.

        Args:
            test_image: 2D numpy array (grayscale face, 100x100)
        Returns:
            dict with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Please train the model first.")

        start_time = time.time()

        x = test_image.astype(np.float32) / 255.0
        x = x.reshape(1, self.img_size[0], self.img_size[1], 1)

        pred_prob = self.model.predict(x, verbose=0)
        pred_class = np.argmax(pred_prob[0])
        confidence = float(pred_prob[0][pred_class]) * 100

        prediction_time = time.time() - start_time
        self.metrics['prediction_speed'] = round(prediction_time, 6)

        predicted_label = self.label_names[pred_class]

        original_b64 = array_to_base64(test_image)

        return {
            'predicted_label': predicted_label,
            'confidence': round(confidence, 2),
            'prediction_time': round(prediction_time, 6),
            'original_image': original_b64,
            'all_probabilities': {
                self.label_names[i]: round(float(pred_prob[0][i]) * 100, 2)
                for i in range(len(self.label_names))
            }
        }
