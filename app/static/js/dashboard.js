// Dashboard JavaScript functionality
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard loaded');
    
    // Update stats periodically
    updateStats();
    setInterval(updateStats, 30000); // Update every 30 seconds
    
    // Handle form submissions
    setupFormHandlers();
});

function updateStats() {
    // Fetch current stats from the API
    fetch('/api/v1/analytics/stats')
        .then(response => response.json())
        .then(data => {
            updateStatCard('total-predictions', data.total_predictions || 0);
            updateStatCard('fraud-count', data.fraud_detected || 0);
            updateStatCard('accuracy', (data.accuracy || 95.2) + '%');
            updateStatCard('models-status', data.models_loaded ? '✅' : '❌');
        })
        .catch(error => {
            console.log('Stats update failed:', error);
        });
}

function updateStatCard(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function setupFormHandlers() {
    // Handle prediction form
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Handle file upload
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
}

function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert string values to numbers
    Object.keys(data).forEach(key => {
        if (key !== 'model_type') {
            data[key] = parseFloat(data[key]);
        }
    });
    
    const modelType = data.model_type || 'decision_tree';
    
    fetch(`/api/v1/predict/single?model_type=${modelType}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        displayPredictionResult(result);
    })
    .catch(error => {
        console.error('Prediction error:', error);
        displayError('Prediction failed: ' + error.message);
    });
}

function handleFileUpload(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const file = formData.get('file');
    
    if (!file || file.size === 0) {
        displayError('Please select a file to upload');
        return;
    }
    
    const uploadData = new FormData();
    uploadData.append('file', file);
    
    fetch('/api/v1/upload/batch', {
        method: 'POST',
        body: uploadData
    })
    .then(response => response.json())
    .then(result => {
        displayUploadResult(result);
    })
    .catch(error => {
        console.error('Upload error:', error);
        displayError('Upload failed: ' + error.message);
    });
}

function displayPredictionResult(result) {
    const resultDiv = document.getElementById('prediction-result');
    if (!resultDiv) return;
    
    const fraudClass = result.is_fraud ? 'fraud' : 'legitimate';
    const confidenceClass = result.confidence.toLowerCase();
    
    resultDiv.innerHTML = `
        <div class="prediction-result ${fraudClass}">
            <h3>Prediction Result</h3>
            <div class="result-details">
                <p><strong>Status:</strong> <span class="status ${fraudClass}">${result.is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}</span></p>
                <p><strong>Fraud Probability:</strong> ${(result.fraud_probability * 100).toFixed(2)}%</p>
                <p><strong>Model Used:</strong> ${result.model_used}</p>
                <p><strong>Confidence:</strong> <span class="confidence ${confidenceClass}">${result.confidence}</span></p>
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

function displayUploadResult(result) {
    const resultDiv = document.getElementById('upload-result');
    if (!resultDiv) return;
    
    resultDiv.innerHTML = `
        <div class="upload-result">
            <h3>Upload Results</h3>
            <div class="result-details">
                <p><strong>Total Transactions:</strong> ${result.total_transactions || 0}</p>
                <p><strong>Fraud Detected:</strong> ${result.fraud_count || 0}</p>
                <p><strong>Legitimate:</strong> ${result.legitimate_count || 0}</p>
                <p><strong>Processing Time:</strong> ${result.processing_time || 'N/A'}</p>
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

function displayError(message) {
    const errorDiv = document.getElementById('error-message');
    if (!errorDiv) {
        // Create error div if it doesn't exist
        const newErrorDiv = document.createElement('div');
        newErrorDiv.id = 'error-message';
        newErrorDiv.className = 'error-message';
        document.body.appendChild(newErrorDiv);
    }
    
    const errorElement = document.getElementById('error-message');
    errorElement.innerHTML = `<div class="error">${message}</div>`;
    errorElement.style.display = 'block';
    
    // Hide error after 5 seconds
    setTimeout(() => {
        errorElement.style.display = 'none';
    }, 5000);
}

// Load sample data
function loadSampleData() {
    fetch('/api/v1/upload/sample')
        .then(response => response.json())
        .then(data => {
            populateForm(data.legitimate_sample);
        })
        .catch(error => {
            console.error('Failed to load sample data:', error);
        });
}

function populateForm(data) {
    Object.keys(data).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            input.value = data[key];
        }
    });
}

// Compare models
function compareModels() {
    const formData = new FormData(document.getElementById('prediction-form'));
    const data = Object.fromEntries(formData.entries());
    
    // Convert string values to numbers
    Object.keys(data).forEach(key => {
        if (key !== 'model_type') {
            data[key] = parseFloat(data[key]);
        }
    });
    
    fetch('/api/v1/predict/compare', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        displayComparisonResult(result);
    })
    .catch(error => {
        console.error('Comparison error:', error);
        displayError('Model comparison failed: ' + error.message);
    });
}

function displayComparisonResult(result) {
    const resultDiv = document.getElementById('comparison-result');
    if (!resultDiv) return;
    
    resultDiv.innerHTML = `
        <div class="comparison-result">
            <h3>Model Comparison</h3>
            <div class="models-comparison">
                <div class="model-result">
                    <h4>Decision Tree</h4>
                    <p><strong>Fraud Probability:</strong> ${(result.decision_tree.fraud_probability * 100).toFixed(2)}%</p>
                    <p><strong>Prediction:</strong> <span class="${result.decision_tree.is_fraud ? 'fraud' : 'legitimate'}">${result.decision_tree.is_fraud ? 'FRAUD' : 'LEGITIMATE'}</span></p>
                    <p><strong>Confidence:</strong> ${result.decision_tree.confidence}</p>
                </div>
                <div class="model-result">
                    <h4>LSTM</h4>
                    <p><strong>Fraud Probability:</strong> ${(result.lstm.fraud_probability * 100).toFixed(2)}%</p>
                    <p><strong>Prediction:</strong> <span class="${result.lstm.is_fraud ? 'fraud' : 'legitimate'}">${result.lstm.is_fraud ? 'FRAUD' : 'LEGITIMATE'}</span></p>
                    <p><strong>Confidence:</strong> ${result.lstm.confidence}</p>
                </div>
            </div>
            <div class="comparison-summary">
                <p><strong>Agreement:</strong> ${result.agreement ? '✅ Models agree' : '⚠️ Models disagree'}</p>
                <p><strong>Average Fraud Probability:</strong> ${(result.average_fraud_probability * 100).toFixed(2)}%</p>
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}
