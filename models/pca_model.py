"""
pca_model.py
PCA (Eigenfaces) face recognition implementation using scikit-learn.
"""
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from utils.visualization import (
    plot_eigenvalue_distribution, plot_confusion_matrix, array_to_base64
)


class PCAFaceRecognizer:
    """Eigenfaces-based face recognition using PCA."""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = None
        self.mean_face = None
        self.eigenfaces = None
        self.train_projections = None
        self.train_labels = None
        self.label_names = None
        self.metrics = {}

    def train(self, images, labels, label_names):
        """
        Train the PCA model on the given face images.

        Args:
            images: list of 2D numpy arrays (grayscale faces)
            labels: list of string labels
            label_names: sorted unique label names
        Returns:
            dict with training results and metrics
        """
        self.label_names = label_names
        start_time = time.time()

        # Flatten images into a data matrix
        n_samples = len(images)
        h, w = images[0].shape
        X = np.array([img.flatten().astype(np.float64) for img in images])
        y = np.array([label_names.index(l) for l in labels])

        # Train/test split
        if n_samples < 4:
            # Too few samples, use all for training
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
            )

        # Compute mean face
        self.mean_face = np.mean(X_train, axis=0)

        # Adjust n_components
        max_components = min(self.n_components, X_train.shape[0], X_train.shape[1])
        self.n_components = max(1, max_components)

        # Fit PCA
        self.pca = PCA(n_components=self.n_components, whiten=True, random_state=42)
        self.pca.fit(X_train)

        # Store eigenfaces
        self.eigenfaces = self.pca.components_

        # Project training data
        self.train_projections = self.pca.transform(X_train)
        self.train_labels = y_train

        # Predict on test set
        test_projections = self.pca.transform(X_test)
        y_pred = []
        for proj in test_projections:
            distances = np.linalg.norm(self.train_projections - proj, axis=1)
            nearest_idx = np.argmin(distances)
            y_pred.append(self.train_labels[nearest_idx])
        y_pred = np.array(y_pred)

        training_time = time.time() - start_time

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Generate visualizations
        mean_face_img = self.mean_face.reshape(h, w)
        mean_face_b64 = array_to_base64(mean_face_img)

        # Top eigenfaces
        n_eigenfaces_display = min(10, self.n_components)
        eigenface_images = []
        for i in range(n_eigenfaces_display):
            ef = self.eigenfaces[i].reshape(h, w)
            eigenface_images.append(array_to_base64(ef))

        # Eigenvalue distribution chart
        eigenvalue_chart = plot_eigenvalue_distribution(
            self.pca.explained_variance_ratio_[:min(20, self.n_components)]
        )

        # Confusion matrix chart
        cm_chart = plot_confusion_matrix(cm, label_names)

        # Dimensionality reduction stats
        original_dim = h * w
        reduced_dim = self.n_components
        total_variance = sum(self.pca.explained_variance_ratio_) * 100

        self.metrics = {
            'accuracy': float(accuracy),
            'training_time': round(training_time, 4),
            'prediction_speed': 0,  # Will be set during predict
            'n_components': self.n_components,
            'original_dim': original_dim,
            'reduced_dim': reduced_dim,
            'variance_retained': round(total_variance, 2),
            'n_train': len(X_train),
            'n_test': len(X_test),
        }

        return {
            'metrics': self.metrics,
            'mean_face': mean_face_b64,
            'eigenfaces': eigenface_images,
            'eigenvalue_chart': eigenvalue_chart,
            'confusion_matrix': cm_chart,
        }

    def predict(self, test_image):
        """
        Predict the identity of a test face image.

        Args:
            test_image: 2D numpy array (grayscale face, 100x100)
        Returns:
            dict with prediction results
        """
        if self.pca is None:
            raise RuntimeError("Model not trained yet. Please train the model first.")

        start_time = time.time()

        h, w = test_image.shape
        x = test_image.flatten().astype(np.float64).reshape(1, -1)

        # Project into eigenspace
        projection = self.pca.transform(x)

        # Find nearest neighbor
        distances = np.linalg.norm(self.train_projections - projection, axis=1)
        nearest_idx = np.argmin(distances)
        min_distance = distances[nearest_idx]
        predicted_label_idx = self.train_labels[nearest_idx]
        predicted_label = self.label_names[predicted_label_idx]

        # Confidence: inverse of distance, normalized
        max_distance = np.max(distances)
        confidence = max(0, (1 - min_distance / max_distance)) * 100

        prediction_time = time.time() - start_time
        self.metrics['prediction_speed'] = round(prediction_time, 6)

        # Reconstruct face from eigenspace
        reconstructed = self.pca.inverse_transform(projection).reshape(h, w)
        original_b64 = array_to_base64(test_image)
        reconstructed_b64 = array_to_base64(reconstructed)

        return {
            'predicted_label': predicted_label,
            'distance': round(float(min_distance), 4),
            'confidence': round(float(confidence), 2),
            'prediction_time': round(prediction_time, 6),
            'original_image': original_b64,
            'reconstructed_image': reconstructed_b64,
        }
