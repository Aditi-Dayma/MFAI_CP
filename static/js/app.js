/**
 * app.js
 * FaceRec AI — Frontend logic for PCA vs CNN Face Recognition.
 */

// ============================================
//  NAVIGATION
// ============================================

function navigateTo(section) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => {
        s.classList.remove('active');
    });

    // Show target section
    const target = document.getElementById('section-' + section);
    if (target) {
        target.classList.add('active');
    }

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    const navLink = document.getElementById('nav-' + section);
    if (navLink) {
        navLink.classList.add('active');
    }
}

// Attach sidebar nav click handlers
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.getAttribute('data-section');
            if (section) {
                navigateTo(section);
            }
        });
    });
});


// ============================================
//  TOAST NOTIFICATIONS
// ============================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 4500);
}


// ============================================
//  DATASET UPLOAD
// ============================================

const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadDataset(e.target.files);
        }
    });
}

// Drag and drop support
if (uploadZone) {
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            uploadDataset(e.dataTransfer.files);
        }
    });
}

async function uploadDataset(files) {
    const progressCard = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const datasetInfo = document.getElementById('datasetInfo');
    const previewCard = document.getElementById('previewCard');

    progressCard.style.display = 'block';
    progressBar.style.width = '10%';
    progressText.textContent = 'Preparing upload...';

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    progressBar.style.width = '30%';
    progressText.textContent = `Uploading ${files.length} files...`;

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        progressBar.style.width = '70%';
        progressText.textContent = 'Processing images...';

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        progressBar.style.width = '100%';
        progressText.textContent = 'Done!';

        // Update dataset info
        const statsGrid = document.getElementById('datasetStats');
        statsGrid.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${data.total_images}</div>
                <div class="stat-label">Total Faces</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.n_classes}</div>
                <div class="stat-label">People</div>
            </div>
            ${Object.entries(data.samples_per_class || {}).map(([name, count]) => `
                <div class="stat-card">
                    <div class="stat-value">${count}</div>
                    <div class="stat-label">${name}</div>
                </div>
            `).join('')}
        `;
        datasetInfo.style.display = 'block';

        // Update preview grid
        if (data.preview_paths && data.preview_paths.length > 0) {
            const previewGrid = document.getElementById('previewGrid');
            previewGrid.innerHTML = data.preview_paths.map(path => `
                <div>
                    <img src="${path}" alt="face preview" loading="lazy">
                </div>
            `).join('');
            previewCard.style.display = 'block';
        }

        // Update home dashboard stats
        document.getElementById('stat-dataset').textContent = data.total_images;
        document.getElementById('stat-classes').textContent = data.n_classes;

        showToast(`Dataset loaded: ${data.total_images} images, ${data.n_classes} classes`, 'success');

    } catch (err) {
        progressBar.style.width = '100%';
        progressBar.style.background = 'var(--accent-red)';
        progressText.textContent = 'Error!';
        showToast(err.message, 'error');
        console.error('Upload error:', err);
    }
}


// ============================================
//  PCA TRAINING
// ============================================

async function trainPCA() {
    const loading = document.getElementById('pcaLoading');
    const results = document.getElementById('pcaResults');
    const placeholder = document.getElementById('pcaPlaceholder');
    const btn = document.getElementById('btnTrainPCA');

    btn.disabled = true;
    loading.style.display = 'flex';
    results.style.display = 'none';
    placeholder.style.display = 'none';

    try {
        const response = await fetch('/api/pca/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_components: 50 }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'PCA training failed');
        }

        loading.style.display = 'none';
        results.style.display = 'block';

        // Render metrics
        const metrics = data.metrics;
        const metricsGrid = document.getElementById('pcaMetrics');
        metricsGrid.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${metrics.training_time}s</div>
                <div class="stat-label">Training Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${metrics.n_components}</div>
                <div class="stat-label">Components</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${metrics.variance_retained}%</div>
                <div class="stat-label">Variance Retained</div>
            </div>
        `;

        // Render mean face
        if (data.mean_face) {
            document.getElementById('meanFaceDisplay').innerHTML =
                `<img src="${data.mean_face}" alt="Mean Face">`;
        }

        // Render eigenfaces
        if (data.eigenfaces && data.eigenfaces.length > 0) {
            document.getElementById('eigenfaceGrid').innerHTML =
                data.eigenfaces.map((ef, i) =>
                    `<img src="${ef}" alt="Eigenface ${i + 1}" title="Eigenface ${i + 1}">`
                ).join('');
        }

        // Render charts
        if (data.eigenvalue_chart) {
            document.getElementById('eigenvalueChart').innerHTML =
                `<img src="${data.eigenvalue_chart}" alt="Eigenvalue Distribution">`;
        }
        if (data.confusion_matrix) {
            document.getElementById('confusionMatrix').innerHTML =
                `<img src="${data.confusion_matrix}" alt="Confusion Matrix">`;
        }

        // Update home stats
        document.getElementById('stat-pca-acc').textContent =
            (metrics.accuracy * 100).toFixed(1) + '%';

        showToast('PCA model trained successfully!', 'success');

    } catch (err) {
        loading.style.display = 'none';
        placeholder.style.display = 'flex';
        showToast(err.message, 'error');
        console.error('PCA training error:', err);
    } finally {
        btn.disabled = false;
    }
}


// ============================================
//  PCA PREDICTION
// ============================================

const pcaTestInput = document.getElementById('pcaTestInput');
let pcaTestFile = null;

if (pcaTestInput) {
    pcaTestInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            pcaTestFile = e.target.files[0];
            document.getElementById('pcaTestFilename').textContent = pcaTestFile.name;
            document.getElementById('btnPredictPCA').disabled = false;
        }
    });
}

async function predictPCA() {
    if (!pcaTestFile) return;

    const btn = document.getElementById('btnPredictPCA');
    btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('image', pcaTestFile);

        const response = await fetch('/api/pca/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'PCA prediction failed');
        }

        const resultDiv = document.getElementById('pcaPredictionResult');
        resultDiv.style.display = 'block';

        // Original image
        if (data.original_image) {
            document.getElementById('pcaOriginal').innerHTML =
                `<img src="${data.original_image}" alt="Original">`;
        }

        // Reconstructed image
        if (data.reconstructed_image) {
            document.getElementById('pcaReconstructed').innerHTML =
                `<img src="${data.reconstructed_image}" alt="Reconstructed">`;
        }

        // Prediction info
        const confClass = data.confidence >= 70 ? 'high' : data.confidence >= 40 ? 'mid' : 'low';
        document.getElementById('pcaPredInfo').innerHTML = `
            <div class="pred-label">${data.predicted_label}</div>
            <div class="pred-detail">
                <span class="detail-label">Confidence</span>
                <span class="detail-value">${data.confidence}%</span>
            </div>
            <div class="pred-detail">
                <span class="detail-label">Distance</span>
                <span class="detail-value">${data.distance}</span>
            </div>
            <div class="pred-detail">
                <span class="detail-label">Prediction Time</span>
                <span class="detail-value">${(data.prediction_time * 1000).toFixed(2)}ms</span>
            </div>
            <div class="confidence-bar-container">
                <div class="confidence-bar-label">
                    <span>Confidence</span>
                    <span>${data.confidence}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill confidence-${confClass}" style="width: ${data.confidence}%"></div>
                </div>
            </div>
        `;

        showToast(`Predicted: ${data.predicted_label} (${data.confidence}%)`, 'success');

    } catch (err) {
        showToast(err.message, 'error');
        console.error('PCA predict error:', err);
    } finally {
        btn.disabled = false;
    }
}


// ============================================
//  CNN TRAINING
// ============================================

async function trainCNN() {
    const loading = document.getElementById('cnnLoading');
    const results = document.getElementById('cnnResults');
    const placeholder = document.getElementById('cnnPlaceholder');
    const btn = document.getElementById('btnTrainCNN');

    btn.disabled = true;
    loading.style.display = 'flex';
    results.style.display = 'none';
    placeholder.style.display = 'none';

    try {
        const response = await fetch('/api/cnn/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs: 20 }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'CNN training failed');
        }

        loading.style.display = 'none';
        results.style.display = 'block';

        // Render metrics
        const metrics = data.metrics;
        const metricsGrid = document.getElementById('cnnMetrics');
        metricsGrid.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
                <div class="stat-label">Test Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${metrics.training_time}s</div>
                <div class="stat-label">Training Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${metrics.epochs}</div>
                <div class="stat-label">Epochs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${metrics.final_train_loss}</div>
                <div class="stat-label">Final Loss</div>
            </div>
        `;

        // Render accuracy/loss chart
        if (data.accuracy_loss_chart) {
            document.getElementById('accLossChart').innerHTML =
                `<img src="${data.accuracy_loss_chart}" alt="Accuracy & Loss">`;
        }

        // Update home stats
        document.getElementById('stat-cnn-acc').textContent =
            (metrics.accuracy * 100).toFixed(1) + '%';

        showToast('CNN model trained successfully!', 'success');

    } catch (err) {
        loading.style.display = 'none';
        placeholder.style.display = 'flex';
        showToast(err.message, 'error');
        console.error('CNN training error:', err);
    } finally {
        btn.disabled = false;
    }
}


// ============================================
//  CNN PREDICTION
// ============================================

const cnnTestInput = document.getElementById('cnnTestInput');
let cnnTestFile = null;

if (cnnTestInput) {
    cnnTestInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            cnnTestFile = e.target.files[0];
            document.getElementById('cnnTestFilename').textContent = cnnTestFile.name;
            document.getElementById('btnPredictCNN').disabled = false;
        }
    });
}

async function predictCNN() {
    if (!cnnTestFile) return;

    const btn = document.getElementById('btnPredictCNN');
    btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('image', cnnTestFile);

        const response = await fetch('/api/cnn/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'CNN prediction failed');
        }

        const resultDiv = document.getElementById('cnnPredictionResult');
        resultDiv.style.display = 'block';

        // Original image
        if (data.original_image) {
            document.getElementById('cnnOriginal').innerHTML =
                `<img src="${data.original_image}" alt="Test Image">`;
        }

        // Prediction info
        const confClass = data.confidence >= 70 ? 'high' : data.confidence >= 40 ? 'mid' : 'low';
        let probsHtml = '';
        if (data.all_probabilities) {
            probsHtml = Object.entries(data.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([label, prob]) => `
                    <div class="pred-detail">
                        <span class="detail-label">${label}</span>
                        <span class="detail-value">${prob}%</span>
                    </div>
                `).join('');
        }

        document.getElementById('cnnPredInfo').innerHTML = `
            <div class="pred-label">${data.predicted_label}</div>
            <div class="pred-detail">
                <span class="detail-label">Confidence</span>
                <span class="detail-value">${data.confidence}%</span>
            </div>
            <div class="pred-detail">
                <span class="detail-label">Prediction Time</span>
                <span class="detail-value">${(data.prediction_time * 1000).toFixed(2)}ms</span>
            </div>
            <div class="confidence-bar-container">
                <div class="confidence-bar-label">
                    <span>Confidence</span>
                    <span>${data.confidence}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill confidence-${confClass}" style="width: ${data.confidence}%"></div>
                </div>
            </div>
            ${probsHtml}
        `;

        showToast(`Predicted: ${data.predicted_label} (${data.confidence}%)`, 'success');

    } catch (err) {
        showToast(err.message, 'error');
        console.error('CNN predict error:', err);
    } finally {
        btn.disabled = false;
    }
}


// ============================================
//  MODEL COMPARISON
// ============================================

async function compareModels() {
    const loading = document.getElementById('compareLoading');
    const results = document.getElementById('compareResults');
    const placeholder = document.getElementById('comparePlaceholder');
    const btn = document.getElementById('btnCompare');

    btn.disabled = true;
    loading.style.display = 'flex';
    results.style.display = 'none';
    placeholder.style.display = 'none';

    try {
        const response = await fetch('/api/compare');
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Comparison failed');
        }

        loading.style.display = 'none';
        results.style.display = 'block';

        const pca = data.pca_metrics || {};
        const cnn = data.cnn_metrics || {};

        // Build comparison table
        const rows = [
            { feature: 'Test Accuracy', pca: pca.accuracy != null ? (pca.accuracy * 100).toFixed(1) + '%' : '—', cnn: cnn.accuracy != null ? (cnn.accuracy * 100).toFixed(1) + '%' : '—' },
            { feature: 'Training Time', pca: pca.training_time != null ? pca.training_time + 's' : '—', cnn: cnn.training_time != null ? cnn.training_time + 's' : '—' },
            { feature: 'Prediction Speed', pca: pca.prediction_speed != null ? (pca.prediction_speed * 1000).toFixed(2) + 'ms' : '—', cnn: cnn.prediction_speed != null ? (cnn.prediction_speed * 1000).toFixed(2) + 'ms' : '—' },
            { feature: 'Training Samples', pca: pca.n_train != null ? pca.n_train : '—', cnn: cnn.n_train != null ? cnn.n_train : '—' },
            { feature: 'Test Samples', pca: pca.n_test != null ? pca.n_test : '—', cnn: cnn.n_test != null ? cnn.n_test : '—' },
            { feature: 'Approach', pca: 'Unsupervised', cnn: 'Supervised' },
            { feature: 'Feature Type', pca: 'Eigenfaces (Linear)', cnn: 'Learned (Nonlinear)' },
        ];

        if (pca.n_components != null) {
            rows.splice(3, 0, { feature: 'PCA Components', pca: pca.n_components, cnn: '—' });
        }
        if (pca.variance_retained != null) {
            rows.splice(4, 0, { feature: 'Variance Retained', pca: pca.variance_retained + '%', cnn: '—' });
        }
        if (cnn.epochs != null) {
            rows.push({ feature: 'Epochs', pca: '—', cnn: cnn.epochs });
        }

        const tbody = document.getElementById('comparisonTableBody');
        tbody.innerHTML = rows.map(row => {
            // Highlight winner for accuracy
            let pcaClass = '';
            let cnnClass = '';
            if (row.feature === 'Test Accuracy' && pca.accuracy != null && cnn.accuracy != null) {
                if (pca.accuracy > cnn.accuracy) pcaClass = 'metric-winner';
                else if (cnn.accuracy > pca.accuracy) cnnClass = 'metric-winner';
            }
            return `
                <tr>
                    <td>${row.feature}</td>
                    <td><span class="metric-value ${pcaClass}">${row.pca}</span></td>
                    <td><span class="metric-value ${cnnClass}">${row.cnn}</span></td>
                </tr>
            `;
        }).join('');

        // Comparison chart
        if (data.comparison_chart) {
            document.getElementById('comparisonChart').innerHTML =
                `<img src="${data.comparison_chart}" alt="Comparison Chart">`;
        }

        // Summary
        const summary = document.getElementById('comparisonSummary');
        if (data.pca_trained && data.cnn_trained) {
            const winner = pca.accuracy >= cnn.accuracy ? 'PCA (Eigenfaces)' : 'CNN';
            const winnerAcc = pca.accuracy >= cnn.accuracy ? pca.accuracy : cnn.accuracy;
            summary.innerHTML = `
                <h3 class="summary-title">📊 Analysis Summary</h3>
                <div class="summary-body">
                    <p>Both models have been trained and evaluated on the same dataset.</p>
                    <p><strong>${winner}</strong> achieved the highest test accuracy of <strong>${(winnerAcc * 100).toFixed(1)}%</strong>.</p>
                    <p>PCA trained in <strong>${pca.training_time}s</strong> while CNN required <strong>${cnn.training_time}s</strong>,
                    demonstrating the computational efficiency trade-off between traditional and deep learning approaches.</p>
                </div>
            `;
        } else {
            const trained = data.pca_trained ? 'PCA' : 'CNN';
            const missing = data.pca_trained ? 'CNN' : 'PCA';
            summary.innerHTML = `
                <h3 class="summary-title">⚠️ Partial Comparison</h3>
                <div class="summary-body">
                    <p>Only the <strong>${trained}</strong> model has been trained. Please train the <strong>${missing}</strong> model for a full comparison.</p>
                </div>
            `;
        }

        showToast('Comparison generated!', 'success');

    } catch (err) {
        loading.style.display = 'none';
        placeholder.style.display = 'flex';
        showToast(err.message, 'error');
        console.error('Compare error:', err);
    } finally {
        btn.disabled = false;
    }
}
