"""
visualization.py
Matplotlib chart generation for PCA and CNN model results.
All charts are saved as PNG files in static/charts/.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import base64

CHART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'charts')


def ensure_chart_dir():
    os.makedirs(CHART_DIR, exist_ok=True)


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#0f1b2d', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def array_to_base64(arr, cmap='gray'):
    """Convert a numpy array (image) to base64 PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(arr, cmap=cmap)
    ax.axis('off')
    fig.patch.set_facecolor('#0f1b2d')
    b64 = fig_to_base64(fig)
    return b64


def plot_eigenvalue_distribution(eigenvalues):
    """Plot eigenvalue distribution as a bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f1b2d')
    ax.set_facecolor('#0f1b2d')

    n = len(eigenvalues)
    bars = ax.bar(range(1, n+1), eigenvalues, color='#4a90d9', edgecolor='#6ab0ff', linewidth=0.5)
    ax.set_xlabel('Principal Component', color='white', fontsize=11)
    ax.set_ylabel('Explained Variance Ratio', color='white', fontsize=11)
    ax.set_title('Eigenvalue Distribution', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#334466')
    ax.spines['left'].set_color('#334466')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, color='white')

    return fig_to_base64(fig)


def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0f1b2d')
    ax.set_facecolor('#0f1b2d')

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha='right', color='white', fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, color='white', fontsize=9)

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=10)

    ax.set_ylabel('True Label', color='white', fontsize=11)
    ax.set_xlabel('Predicted Label', color='white', fontsize=11)
    ax.set_title('Confusion Matrix', color='white', fontsize=13, fontweight='bold')

    return fig_to_base64(fig)


def plot_accuracy_loss(history):
    """Plot CNN training accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f1b2d')

    epochs = range(1, len(history['accuracy']) + 1)

    # Accuracy
    ax1.set_facecolor('#0f1b2d')
    ax1.plot(epochs, history['accuracy'], '#4a90d9', linewidth=2, label='Train Accuracy', marker='o', markersize=3)
    if 'val_accuracy' in history:
        ax1.plot(epochs, history['val_accuracy'], '#ff6b6b', linewidth=2, label='Val Accuracy', marker='s', markersize=3)
    ax1.set_title('Model Accuracy', color='white', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', color='white', fontsize=11)
    ax1.set_ylabel('Accuracy', color='white', fontsize=11)
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a2744', edgecolor='#334466', labelcolor='white')
    ax1.spines['bottom'].set_color('#334466')
    ax1.spines['left'].set_color('#334466')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.2, color='white')

    # Loss
    ax2.set_facecolor('#0f1b2d')
    ax2.plot(epochs, history['loss'], '#4a90d9', linewidth=2, label='Train Loss', marker='o', markersize=3)
    if 'val_loss' in history:
        ax2.plot(epochs, history['val_loss'], '#ff6b6b', linewidth=2, label='Val Loss', marker='s', markersize=3)
    ax2.set_title('Model Loss', color='white', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', color='white', fontsize=11)
    ax2.set_ylabel('Loss', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1a2744', edgecolor='#334466', labelcolor='white')
    ax2.spines['bottom'].set_color('#334466')
    ax2.spines['left'].set_color('#334466')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.2, color='white')

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_comparison_bar(pca_metrics, cnn_metrics):
    """Plot PCA vs CNN comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1b2d')
    ax.set_facecolor('#0f1b2d')

    categories = ['Accuracy (%)', 'Training Time (s)', 'Prediction Speed (ms)']
    pca_vals = [
        pca_metrics.get('accuracy', 0) * 100,
        pca_metrics.get('training_time', 0),
        pca_metrics.get('prediction_speed', 0) * 1000,
    ]
    cnn_vals = [
        cnn_metrics.get('accuracy', 0) * 100,
        cnn_metrics.get('training_time', 0),
        cnn_metrics.get('prediction_speed', 0) * 1000,
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, pca_vals, width, label='PCA (Eigenfaces)',
                   color='#4a90d9', edgecolor='#6ab0ff', linewidth=0.5)
    bars2 = ax.bar(x + width/2, cnn_vals, width, label='CNN',
                   color='#ff6b6b', edgecolor='#ff8888', linewidth=0.5)

    ax.set_xlabel('Metric', color='white', fontsize=11)
    ax.set_ylabel('Value', color='white', fontsize=11)
    ax.set_title('PCA vs CNN Comparison', color='white', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a2744', edgecolor='#334466', labelcolor='white', fontsize=10)
    ax.spines['bottom'].set_color('#334466')
    ax.spines['left'].set_color('#334466')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, color='white')

    # Value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    color='white', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    color='white', fontsize=9)

    plt.tight_layout()
    return fig_to_base64(fig)
