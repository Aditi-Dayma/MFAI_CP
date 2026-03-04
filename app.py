"""
app.py
Flask application — REST API for PCA vs CNN Face Recognition.
"""
import os
import sys
import json
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import process_uploaded_images, process_single_image
from utils.visualization import plot_comparison_bar
from models.pca_model import PCAFaceRecognizer
from models.cnn_model import CNNFaceRecognizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'charts')

# Global model instances
pca_model = PCAFaceRecognizer(n_components=50)
cnn_model = CNNFaceRecognizer()

# Global dataset cache
dataset_cache = {
    'images': [],
    'labels': [],
    'label_names': [],
    'preview_paths': [],
    'loaded': False
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Upload face images organized by person (folder structure)."""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        # Clear previous uploads
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Save uploaded files preserving folder structure
        saved_count = 0
        for f in files:
            if not f.filename:
                continue
            # The relative path from the webkitdirectory upload
            rel_path = f.filename.replace('\\', '/')
            parts = rel_path.split('/')

            # We expect: topfolder/person_name/image.jpg
            # or just: person_name/image.jpg
            if len(parts) >= 2:
                # Use the second-to-last part as label, last as filename
                label = parts[-2]
                fname = parts[-1]
            else:
                # Single file without folder structure — skip
                continue

            ext = fname.lower().split('.')[-1]
            if ext not in ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'):
                continue

            save_dir = os.path.join(UPLOAD_DIR, label)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, fname)
            f.save(save_path)
            saved_count += 1

        if saved_count == 0:
            return jsonify({'error': 'No valid image files found. Please upload a folder with subfolders for each person.'}), 400

        # Process images
        images, labels, label_names, preview_paths = process_uploaded_images(UPLOAD_DIR)

        if len(images) == 0:
            return jsonify({'error': 'No faces could be detected in the uploaded images.'}), 400

        # Cache dataset
        dataset_cache['images'] = images
        dataset_cache['labels'] = labels
        dataset_cache['label_names'] = label_names
        dataset_cache['preview_paths'] = preview_paths
        dataset_cache['loaded'] = True

        return jsonify({
            'success': True,
            'total_images': len(images),
            'n_classes': len(label_names),
            'label_names': label_names,
            'samples_per_class': {
                name: labels.count(name) for name in label_names
            },
            'preview_paths': preview_paths[:30],  # Limit preview
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pca/train', methods=['POST'])
def train_pca():
    """Train the PCA (Eigenfaces) model."""
    try:
        if not dataset_cache['loaded']:
            return jsonify({'error': 'No dataset loaded. Please upload images first.'}), 400

        n_components = request.json.get('n_components', 50) if request.is_json else 50

        global pca_model
        pca_model = PCAFaceRecognizer(n_components=n_components)

        results = pca_model.train(
            dataset_cache['images'],
            dataset_cache['labels'],
            dataset_cache['label_names']
        )

        return jsonify({
            'success': True,
            **results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pca/predict', methods=['POST'])
def predict_pca():
    """Predict using the PCA model."""
    try:
        if pca_model.pca is None:
            return jsonify({'error': 'PCA model not trained yet. Please train the model first.'}), 400

        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No test image provided.'}), 400

        test_image = process_single_image(file.read())
        result = pca_model.predict(test_image)

        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cnn/train', methods=['POST'])
def train_cnn():
    """Train the CNN model."""
    try:
        if not dataset_cache['loaded']:
            return jsonify({'error': 'No dataset loaded. Please upload images first.'}), 400

        epochs = request.json.get('epochs', 20) if request.is_json else 20

        global cnn_model
        cnn_model = CNNFaceRecognizer()

        results = cnn_model.train(
            dataset_cache['images'],
            dataset_cache['labels'],
            dataset_cache['label_names'],
            epochs=epochs
        )

        return jsonify({
            'success': True,
            **results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cnn/predict', methods=['POST'])
def predict_cnn():
    """Predict using the CNN model."""
    try:
        if cnn_model.model is None:
            return jsonify({'error': 'CNN model not trained yet. Please train the model first.'}), 400

        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No test image provided.'}), 400

        test_image = process_single_image(file.read())
        result = cnn_model.predict(test_image)

        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['GET'])
def compare_models():
    """Compare PCA and CNN model metrics."""
    try:
        pca_trained = pca_model.pca is not None
        cnn_trained = cnn_model.model is not None

        if not pca_trained and not cnn_trained:
            return jsonify({'error': 'Neither model has been trained. Please train at least one model.'}), 400

        pca_metrics = pca_model.metrics if pca_trained else {}
        cnn_metrics = cnn_model.metrics if cnn_trained else {}

        # Generate comparison chart
        comparison_chart = None
        if pca_trained and cnn_trained:
            comparison_chart = plot_comparison_bar(pca_metrics, cnn_metrics)

        return jsonify({
            'success': True,
            'pca_trained': pca_trained,
            'cnn_trained': cnn_trained,
            'pca_metrics': pca_metrics,
            'cnn_metrics': cnn_metrics,
            'comparison_chart': comparison_chart,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
