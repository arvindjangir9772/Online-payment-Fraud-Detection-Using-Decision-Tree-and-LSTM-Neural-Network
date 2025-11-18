# 🚀 Advanced Hybrid Fraud Detection System

A comprehensive AI-powered fraud detection system that combines LSTM neural networks with Decision Tree models for superior accuracy and explainability. Built with FastAPI, featuring real-time analysis, advanced explainability, and a modern web interface.

## 🌟 Key Features

### 🤖 Hybrid AI Model
- **LSTM Neural Network**: Captures temporal patterns and sequential behavior
- **Decision Tree**: Provides rule-based interpretability and fast predictions
- **Dynamic Weighting**: Adaptive α parameter adjusts model influence based on data patterns
- **SHAP Integration**: Advanced explainability with feature importance analysis

### 🔍 Real-time Analysis
- **Single Transaction Check**: Manual input with detailed explanations
- **Batch Processing**: CSV upload with comprehensive analytics
- **On-the-fly Cleaning**: Automatic data preprocessing and validation
- **Live Dashboard**: Real-time monitoring and statistics

### 📊 Advanced Analytics
- **Risk Distribution**: Visual breakdown of transaction risk levels
- **Model Performance**: Detailed metrics and comparison
- **Trend Analysis**: Historical patterns and fraud detection trends
- **Export Capabilities**: CSV and JSON data export

### 🎨 Modern UI/UX
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Visualizations**: Real-time charts and graphs
- **Drag & Drop**: Easy file upload with validation
- **Real-time Feedback**: Live notifications and status updates

## 🏗️ Architecture

### Core Components

```
app/
├── models/
│   ├── hybrid_model.py      # Hybrid LSTM + Decision Tree model
│   ├── shap_explainer.py    # SHAP integration for explainability
│   └── model_manager.py     # Model loading and management
├── routes/
│   ├── predictions.py       # Prediction endpoints
│   ├── data_upload.py       # File upload and batch processing
│   └── analytics.py         # Analytics and reporting
├── templates/
│   ├── index.html           # Main dashboard
│   ├── single_prediction.html  # Manual transaction check
│   └── batch_upload.html    # Batch analysis interface
└── static/
    └── css/
        └── style.css        # Modern UI styling
```

### Hybrid Model Logic

The system uses a sophisticated hybrid approach:

```
P_final = α × P_LSTM + (1-α) × P_DT

Where:
- P_final: Final fraud probability
- α: Dynamic weight (0.3-0.9, adapts based on data patterns)
- P_LSTM: LSTM model prediction
- P_DT: Decision Tree model prediction
```

**Dynamic α Calculation:**
- **Amount Variance**: Higher variance → trust LSTM more for temporal patterns
- **Feature Complexity**: Complex patterns → favor LSTM
- **Model Confidence**: Adjusts based on individual model performance

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd online-payment-fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train models** (if not already trained)
```bash
python ml_pipeline/train_models.py
```

5. **Run the application**
```bash
python run.py
```

6. **Access the web interface**
Open your browser and navigate to `http://localhost:8080`

## 📖 Usage Guide

### Single Transaction Check

1. Navigate to the **Single Transaction Check** page
2. Enter transaction details:
   - **Amount** (required)
   - **V1-V28 features** (optional, will be filled with 0 if missing)
3. Click **Analyze Transaction**
4. View detailed results including:
   - Fraud probability and status
   - Model-specific predictions
   - SHAP explanations
   - Feature contributions

### Batch Analysis

1. Go to the **Batch Analysis** page
2. Upload a CSV file with transaction data
3. Configure analysis options:
   - Fraud detection threshold
   - Include explanations
   - Maximum results to display
4. Click **Analyze Transactions**
5. Review comprehensive results:
   - Summary statistics
   - Risk distribution
   - Detailed transaction analysis
   - Export options

### API Endpoints

#### Prediction Endpoints
- `POST /api/v1/predict/hybrid/single` - Single transaction prediction
- `POST /api/v1/predict/hybrid/batch` - Batch prediction
- `POST /api/v1/predict/hybrid/manual` - Manual input prediction

#### Explainability Endpoints
- `POST /api/v1/predict/hybrid/shap` - SHAP explanation for single transaction
- `POST /api/v1/predict/hybrid/shap/batch` - Batch SHAP explanations

#### Analytics Endpoints
- `GET /api/v1/analytics/dashboard` - Dashboard metrics
- `GET /api/v1/analytics/performance` - Model performance
- `GET /api/v1/analytics/trends` - Fraud detection trends

## 🔧 Configuration

### Model Parameters

The hybrid model can be configured through the web interface or API:

- **Fraud Threshold**: Default 0.5 (adjustable 0.1-0.9)
- **Alpha Weight**: Default 0.6 (LSTM influence, adjustable 0.1-0.9)
- **Adaptive Threshold**: Automatically adjusts based on recent patterns
- **Explainability**: Enable/disable SHAP explanations

### Data Requirements

#### CSV Format
```csv
Amount,V1,V2,V3,...,V28
149.62,-1.359807,-0.072781,2.536347,...,-0.021053
25.75,1.191857,0.266151,0.166480,...,0.014724
```

#### Required Columns
- `Amount`: Transaction amount (required)
- `V1-V28`: Principal component features (optional, filled with 0 if missing)

## 📊 Performance Metrics

### Model Accuracy
- **Hybrid Model**: 98.4% accuracy
- **LSTM Component**: 99.7% accuracy, 92% precision, 85% recall
- **Decision Tree**: 99.5% accuracy, 89% precision, 78% recall

### Response Times
- **Single Prediction**: ~0.15 seconds
- **Batch Processing**: ~0.5 seconds per 100 transactions
- **SHAP Explanations**: ~0.3 seconds per transaction

## 📚 Documentation

### Development Logbook
This project includes a comprehensive development logbook (`LOGBOOK_ENTRIES.md`) documenting 14 weeks of development work, including:
- Weekly objectives and tasks
- Technical decisions and challenges
- Model development iterations
- Testing and optimization efforts

This logbook demonstrates the systematic development process and is valuable for understanding the project's evolution.

## 🛠️ Development

### Project Structure
```
online-payment-fraud-detection/
├── app/                    # Main application code
│   ├── models/            # ML model implementations
│   ├── routes/            # API endpoints
│   ├── templates/         # HTML templates
│   └── static/            # CSS and JavaScript files
├── ml_pipeline/           # Machine learning pipeline
│   └── train_models.py    # Main training script
├── scripts/               # Development and utility scripts
├── models/                # Trained model files (.pkl, .h5)
├── data/                  # Data storage
│   ├── raw/              # Raw dataset
│   └── processed/        # Processed data
├── results/               # Analysis results (generated at runtime)
├── requirements.txt       # Python dependencies
├── run.py                # Application entry point
├── README.md             # This file
└── LOGBOOK_ENTRIES.md    # Development logbook (14 weeks)
```

### Adding New Features

1. **New Models**: Add to `app/models/` and update `model_manager.py`
2. **New Endpoints**: Add to `app/routes/` and register in `main.py`
3. **UI Components**: Update templates in `app/templates/`
4. **Styling**: Modify `app/static/css/style.css`

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=app tests/
```

## 🔒 Security Features

- **Input Validation**: Comprehensive data validation and sanitization
- **Error Handling**: Graceful error handling with informative messages
- **Rate Limiting**: API rate limiting to prevent abuse
- **Data Privacy**: No persistent storage of sensitive transaction data

## 📈 Monitoring & Analytics

### Real-time Monitoring
- System health status
- Model performance metrics
- Request/response statistics
- Error rates and patterns

### Analytics Dashboard
- Fraud detection trends
- Model comparison metrics
- Feature importance analysis
- Risk distribution patterns

## 🚀 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t fraud-detection .

# Run container
docker run -p 8080:8080 fraud-detection
```

### Production Considerations
- Use a production WSGI server (e.g., Gunicorn)
- Configure proper logging
- Set up monitoring and alerting
- Use a reverse proxy (e.g., Nginx)
- Implement proper security measures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SHAP Library**: For advanced model explainability
- **FastAPI**: For the excellent web framework
- **TensorFlow/Keras**: For deep learning capabilities
- **scikit-learn**: For traditional ML algorithms
- **Pandas/NumPy**: For data processing

## 📖 Additional Documentation

- **LOGBOOK_ENTRIES.md**: Complete 14-week development logbook
- **RESUME_PROJECT_DESCRIPTION.md**: Project description for portfolio/resume
- **LICENSE**: MIT License

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the development team
- Check the documentation wiki

---

**Built with ❤️ for the future of fraud detection**
