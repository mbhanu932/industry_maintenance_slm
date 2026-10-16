# Industrial Maintenance Copilot - AI-Powered Predictive Maintenance System

## Overview

**Industrial Maintenance Copilot** is an advanced AI-powered system that combines Small Language Models (SLMs) with machine learning to provide real-time, data-driven maintenance recommendations for industrial equipment. The system analyzes engine telemetry, predicts equipment degradation, and provides actionable insights to prevent costly failures.

### Key Features

✨ **AI-Powered Analysis**
- Retrieval-Augmented Generation (RAG) for contextual maintenance knowledge
- Fine-tuned SLM for industrial-specific insights
- Safety guardrails for evidence-based recommendations

🔮 **Predictive Maintenance**
- Remaining Useful Life (RUL) prediction with ML models
- Real-time degradation monitoring
- Proactive failure prediction

📊 **Real-Time Monitoring**
- Fleet-wide heat map visualization
- Priority-based maintenance board
- Historical trend analysis

🎯 **Actionable Insights**
- Root cause analysis with AI investigation
- Maintenance SOP recommendations
- Risk scoring and prioritization

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Transformers (HuggingFace)** - SLM inference
- **PyTorch** - ML models for RUL prediction
- **Pandas** - Data processing
- **Uvicorn** - ASGI server

### Frontend
- **React** - UI framework
- **TypeScript** - Type-safe development
- **Vite** - Build tool
- **TanStack Query** - API state management
- **Tailwind CSS** - Styling

### Data
- **C-MAPSS Dataset** - 20,631 engine degradation records
- **Parquet Format** - Efficient columnar storage
- **Time Series Data** - Real-time telemetry


## Quick Start

### Local Development (5 minutes)

#### Backend
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend available at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

#### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend available at: `http://localhost:5173`

### Docker (1 command)

```bash
docker-compose up -d
```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Backend Docs: `http://localhost:8000/docs`

## API Endpoints

### Core Endpoints

**Health & Status**
- `GET /health` - Health check
- `GET /status` - System status

**Maintenance Analysis**
- `POST /api/analyze` - Analyze equipment condition
- `POST /api/investigate` - Deep investigation
- `GET /api/recommendations/{engine_id}` - Get recommendations

**Data & Monitoring**
- `GET /api/fleet-status` - Fleet-wide status
- `GET /api/engine/{engine_id}` - Engine details
- `GET /api/metrics` - System metrics

**Authentication** (if enabled)
- `POST /auth/login` - User authentication
- `POST /auth/logout` - Logout

See full API documentation at `/docs` endpoint.

## Core Features Explained

### 1. Predictive Maintenance
- **RUL Prediction**: Machine learning models trained on C-MAPSS dataset
- **Degradation Tracking**: Real-time monitoring of equipment condition
- **Failure Prevention**: Proactive alerts before critical failures

### 2. AI-Powered Insights
- **Root Cause Analysis**: Investigator engine identifies failure modes
- **Knowledge Integration**: RAG system retrieves relevant SOPs
- **Safe Recommendations**: Guardrails ensure evidence-based advice

### 3. Real-Time Visualization
- **Fleet Heat Map**: Color-coded engine status by risk
- **Priority Board**: Sorted maintenance queue
- **Historical Trends**: Degradation patterns over time

### 4. Enterprise-Ready
- **Audit Trail**: Complete operation logging
- **Authentication**: Secure API access
- **Monitoring**: System health and performance metrics

## Data

The system uses the **C-MAPSS Turbofan Engine Degradation Dataset**:
- **20,631 engine cycles** across 100 engines
- **26 sensor readings** per cycle
- **4 operating conditions** (FD001-FD004)
- **Train/Test splits** with RUL labels

Data stored in: `backend/data/processed/`
- Parquet format for efficiency
- Ready for immediate inference

## Models

- **SLM Fine-Tuning**: LoRA adapters in `backend/models/industrial_slm_model_v2/`
- **RUL Prediction**: PyTorch models for each dataset
- **Total Parameters**: ~1B (SLM) + RUL predictors

## Getting Started

1. **Clone repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/industrial-maintenance-copilot.git
   cd industrial-maintenance-copilot
   ```

2. **Choose deployment method**
   - Local: Run backend and frontend separately
   - Docker: `docker-compose up -d`
   - Cloud: Follow DEPLOYMENT_GUIDE.md

3. **Access the system**
   - Frontend: `http://localhost:5173`
   - Backend API: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

4. **Explore features**
   - Check fleet status
   - Analyze engine conditions
   - Get maintenance recommendations

## Configuration

### Environment Variables

**Backend** (`.env`)
```
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=./models/industrial_slm_model_v2
DATA_PATH=./data/processed
KB_PATH=./knowledge_base/documents
```

**Frontend** (`.env.local`)
```
VITE_API_BASE_URL=http://localhost:8000
```

See `.env.example` files in backend/ and frontend/ directories.

## Deployment

### Quick Deployment

1. **GitHub Setup**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo>
   git push -u origin main
   ```

2. **Docker Deployment**
   ```bash
   docker-compose up -d
   ```

3. **Cloud Platforms**
   - AWS: ECS/EC2 + S3 + CloudFront
   - Heroku: Simple git push deployment
   - DigitalOcean: Docker VM
   - Kubernetes: Full orchestration

See **DEPLOYMENT_GUIDE.md** for detailed instructions.

## Performance

- **API Response Time**: <500ms (average)
- **Frontend Build Size**: 33.67 KB JS + 17.04 KB CSS
- **Model Inference**: <200ms per request
- **Data Processing**: 20,631 records in <1s

## Testing

Run the test suite (if enabled):
```bash
pytest
```

For Docker:
```bash
docker-compose exec backend pytest
```

## Documentation

Comprehensive documentation available in `/docs`:

1. **01_PROJECT_SUMMARY.md** - Quick overview
2. **02_UI_WALKTHROUGH.md** - How to use the system
3. **03_DEMO_SCRIPT.md** - Live demo narrative
4. **04_TECHNICAL_IMPLEMENTATION.md** - Technical deep dive
5. **05_PROJECT_DEFINITION_AND_ARCHITECTURE.md** - System architecture
6. **06_SIMPLE_EXPLANATION.md** - Non-technical overview
7. **07_TECH_STACK_SUMMARY.md** - Technology details
8. **08_DETAILED_PROJECT_SUMMARY.md** - Comprehensive guide (5000+ words)
9. **09_THEORETICAL_PROJECT_SUMMARY.md** - Academic perspective

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React)                     │
│                    [UI Components]                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ HTTP/REST
                       ↓
┌─────────────────────────────────────────────────────────┐
│                  Backend (FastAPI)                       │
│  ┌────────────────────────────────────────────────────┐ │
│  │  API Routes & Authentication                       │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Inference Engine                                  │ │
│  │  ├─ SLM (Small Language Model)                    │ │
│  │  ├─ RAG (Retrieval-Augmented Generation)         │ │
│  │  ├─ Investigator (Root Cause Analysis)           │ │
│  │  └─ Model Guardrails (Safety Constraints)        │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  ML Models                                         │ │
│  │  ├─ RUL Prediction (PyTorch)                      │ │
│  │  └─ Degradation Analysis                          │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Data Layer                                        │ │
│  │  ├─ C-MAPSS Dataset (20,631 records)             │ │
│  │  ├─ Knowledge Base (SOPs)                        │ │
│  │  └─ Telemetry Storage                            │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request


## Acknowledgments

- **Data Source**: NASA's C-MAPSS Turbofan Engine Degradation Dataset
- **AI Models**: HuggingFace Transformers
- **Frontend Framework**: React + Vite
- **Backend Framework**: FastAPI

---

**Ready to deploy?** See **DEPLOYMENT_GUIDE.md** for step-by-step instructions.

**Want to understand the system?** Start with **docs/01_PROJECT_SUMMARY.md**.

**Building on this?** Check **docs/05_PROJECT_DEFINITION_AND_ARCHITECTURE.md** for system design.

---
