# 🇹🇿 TSL (Tanzania Sign Language) Detection System

A real-time sign language gesture recognition system using MediaPipe hand detection and a Transformer deep learning model. Supports 12 different gestures with bilingual output (English & Swahili).

## 🎯 Features

- ✅ **Real-time gesture detection** via webcam
- ✅ **12 gesture classes** (Hello, Stop, One, Peace, Three, Four, Good, I Love You, Promise, L-Shape, Rock On, Unknown)
- ✅ **Bilingual output** (English + Swahili translations)
- ✅ **Transformer-based model** with MediaPipe hand landmarks
- ✅ **WebSocket streaming** for low-latency detection
- ✅ **Modern UI** built with Next.js 15 and React 19
- ✅ **Text-to-speech** for accessibility
- ✅ **Free GPU training** via Google Colab

## 📋 System Architecture

```
Frontend (Next.js 15)          Backend (FastAPI)
├── Camera View                ├── MediaPipe Hand Detection
├── Real-time rendering        ├── Transformer Model Inference
├── Gesture history (50)        └── WebSocket /ws/detect
└── Bilingual output
                               Database/Storage
                               └── Checkpoints & Models
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (backend)
- Node.js 18+ (frontend)
- Webcam (for real-time detection)
- 500MB free disk space

### 1. Clone & Setup

```bash
# Clone the project
git clone https://github.com/your-username/TSL-System.git
cd TSL-System

# Create Python virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Mac/Linux

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### 2. Train the Model (Choose One)

#### Option A: Use Google Colab (Recommended - Free GPU) 🎉

```bash
# Fastest option with free GPU acceleration
# See: COLAB_TRAINING.md for detailed instructions
```

#### Option B: Train Locally

```bash
python train.py
# This generates synthetic data and trains the model (~1-2 min on CPU)
```

### 3. Start the Backend

```bash
cd backend
python main.py
# Backend runs on http://localhost:8000
# Health check: http://localhost:8000/health
```

### 4. Start the Frontend

```bash
# In a new terminal
cd frontend
npm install  # First time only
npm run dev
# Frontend runs on http://localhost:3002
```

### 5. Open in Browser

Visit: **http://localhost:3002**

Click **"Start Detection"** and make some gestures to see the model in action!

---

## 🎓 Training Your Own Model

### Quick Option: Google Colab (Recommended)

1. Open the Colab notebook: `TSL_Colab_Training.ipynb`
2. Click "Run all"
3. Download trained checkpoint from Google Drive
4. Place in `backend/checkpoints/tsl_transformer.pt`

**See [COLAB_TRAINING.md](./COLAB_TRAINING.md) for detailed instructions**

### Local Training

```bash
cd backend
python train.py
```

Configuration in `train.py`:

- **EPOCHS**: 50 (training iterations)
- **BATCH_SIZE**: 32
- **LR**: 1e-3 (learning rate)
- **Data**: Automatically generates 1,000 synthetic samples

---

## 📁 Project Structure

```
TSL-System/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── train.py                # Training script
│   ├── requirements.txt         # Python dependencies
│   ├── models/
│   │   └── transformer_model.py # Model architecture
│   ├── services/
│   │   ├── ai_service.py       # Inference wrapper
│   │   ├── model_service.py    # Model loader
│   │   ├── mediapipe_service.py# Hand detection
│   │   └── sequence_service.py # Frame buffering
│   ├── routes/
│   │   └── predict.py          # WebSocket endpoint
│   ├── schemas/
│   │   └── prediction_schema.py# Data models
│   ├── utils/
│   │   ├── dataset.py          # Data generation
│   │   └── preprocessing.py    # Frame preprocessing
│   ├── checkpoints/            # Model checkpoints
│   └── hand_landmarker.task    # MediaPipe model
│
├── frontend/
│   ├── package.json            # Node dependencies
│   ├── tsconfig.json           # TypeScript config
│   ├── next.config.ts          # Next.js config
│   ├── app/
│   │   ├── page.tsx            # Main page
│   │   ├── layout.tsx          # Root layout
│   │   └── globals.css         # Global styles
│   ├── components/
│   │   ├── CameraView.tsx      # Webcam + overlay
│   │   ├── Header.tsx          # Navigation
│   │   └── TranslationPanel.tsx# Results display
│   ├── lib/
│   │   ├── api.ts             # Backend API client
│   │   ├── tts.ts             # Text-to-speech
│   │   └── types.ts           # TypeScript types
│   └── public/                 # Static files
│
├── COLAB_TRAINING.md           # Google Colab guide
└── README.md                   # This file
```

---

## 🔧 Configuration

### Backend Configuration

**CORS Allowed Origins** (in `backend/main.py`):

```python
allow_origins=[
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002"
]
```

### Frontend Configuration

**Backend URL** (in `frontend/lib/api.ts`):

```typescript
const API_BASE = "http://localhost:8000";
```

---

## 🤖 Model Details

### Architecture

- **Type**: Transformer Encoder
- **Input**: 126-dimensional hand landmark vectors
  - 2 hands × 21 landmarks × 3 coordinates (x, y, z)
- **Processing**:
  - Positional encoding
  - 4 transformer encoder layers
  - 4 attention heads
  - 128 model dimensions
  - GELU activation
- **Output**: 12-class classification

### Performance

- **Inference Speed**: ~30-50ms per frame on GPU
- **Accuracy**: ~85-95% on synthetic data
- **Model Size**: ~5-10 MB

### Gesture Classes

| Index | Gesture    | English      | Swahili           |
| ----- | ---------- | ------------ | ----------------- |
| 0     | Hello      | "Hello"      | "Habari"          |
| 1     | Stop       | "Stop"       | "Simama"          |
| 2     | One        | "One"        | "Moja"            |
| 3     | Peace      | "Peace"      | "Amani"           |
| 4     | Three      | "Three"      | "Tatu"            |
| 5     | Four       | "Four"       | "Nne"             |
| 6     | Good       | "Good"       | "Nzuri"           |
| 7     | I Love You | "I Love You" | "Nakupenda"       |
| 8     | Promise    | "Promise"    | "Ahidi"           |
| 9     | L-Shape    | "L-Shape"    | "L-Umbo"          |
| 10    | Rock On    | "Rock On"    | "Kujipiga Magoti" |
| 11    | Unknown    | "Unknown"    | "Haijulikana"     |

---

## 🐛 Troubleshooting

### Backend Issues

| Error                      | Solution                                               |
| -------------------------- | ------------------------------------------------------ |
| Port 8000 already in use   | `lsof -i :8000` to find process, or use different port |
| "No module named torch"    | Run `pip install torch` in virtual environment         |
| Health check fails         | Make sure backend is running on port 8000              |
| WebSocket connection fails | Check CORS configuration, enable ports 3000-3002       |

### Frontend Issues

| Error                      | Solution                                            |
| -------------------------- | --------------------------------------------------- |
| "Port 3000 already in use" | Frontend auto-uses 3001, 3002, etc.                 |
| CSS module errors          | Delete `.next` folder: `rm -rf .next`               |
| Blank page                 | Check browser console, verify backend connection    |
| Camera not working         | Grant camera permissions, ensure HTTPS or localhost |

### Model Issues

| Error                    | Solution                                           |
| ------------------------ | -------------------------------------------------- |
| "No checkpoint found"    | Run `python train.py` or use Google Colab training |
| Poor gesture recognition | Train longer, use more diverse data                |
| Memory errors            | Reduce `BATCH_SIZE` in train.py                    |

---

## 📊 Monitoring

### Backend Health Check

```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "timestamp": "2024-01-01T12:00:00Z"}
```

### Training Metrics

View in terminal during training:

```
Epoch  1 | Train Loss: 2.4521 | Train Acc: 0.125 | Val Loss: 2.4234 | Val Acc: 0.150
Epoch  5 | Train Loss: 1.8934 | Train Acc: 0.380 | Val Loss: 1.7823 | Val Acc: 0.420
Epoch 10 | Train Loss: 1.2456 | Train Acc: 0.680 | Val Loss: 1.1234 | Val Acc: 0.710
```

---

## 🤝 Contributing

Contributions welcome! To improve the model:

1. **Collect real data**: Record actual sign language gestures
2. **Fine-tune**: Use transfer learning on the trained checkpoint
3. **Add gestures**: Expand the 12-class model
4. **Optimize**: Reduce model size for mobile deployment

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🎬 Get Started Now!

```bash
# 1. Train on Colab (5 minutes)
# Open: TSL_Colab_Training.ipynb in Google Colab
# Download checkpoint when done

# 2. Or train locally (2 minutes)
cd backend && python train.py

# 3. Start backend
python main.py

# 4. Start frontend (new terminal)
cd frontend && npm run dev

# 5. Open browser
# http://localhost:3002
```

**That's it! Your real-time sign language detector is ready.** 🎉

---

## 📞 Support

- **Issues**: Report on GitHub Issues
- **Questions**: Check existing discussions
- **Model ideas**: See contribution guide above

**Happy gesture detecting!** 🖐️✋👋
