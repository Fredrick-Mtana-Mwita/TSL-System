# 🎓 Training TSL Model on Google Colab

This guide shows how to train the TSL (Tanzania Sign Language) Transformer model using free GPU acceleration on Google Colab.

## Why Google Colab?

✅ **Free GPU access** - Train 10x faster than CPU  
✅ **No setup required** - Just upload and run  
✅ **Google Drive integration** - Automatic checkpoint saving  
✅ **Easy download** - Get trained model directly to your machine

## Quick Start (2 minutes)

### 1. Open Google Colab

Go to: https://colab.research.google.com

### 2. Upload the Notebook

- Click **File** → **Upload notebook**
- Select `TSL_Colab_Training.ipynb` from this project
- Or paste this URL directly in Colab (if on GitHub):
  ```
  https://github.com/YOUR_USERNAME/TSL-System/blob/main/TSL_Colab_Training.ipynb
  ```

### 3. Run Training

- Click **Runtime** → **Run all**
- Watch the training progress (takes ~1-2 minutes with GPU)
- Model saves automatically to Google Drive

### 4. Download Your Model

1. Open [Google Drive](https://drive.google.com/drive/my-drive)
2. Go to folder: `TSL-Checkpoints/`
3. Download `tsl_transformer.pt`
4. Move to local folder: `backend/checkpoints/`

### 5. Use Your Trained Model

```bash
# Terminal 1: Start backend
cd backend
python main.py

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Visit `http://localhost:3002` and click "Start Detection" to use your trained model! 🎉

---

## What's in the Notebook?

| Step | What it does                              | Time |
| ---- | ----------------------------------------- | ---- |
| 1    | Set up Colab, mount Google Drive          | 30s  |
| 2    | Install dependencies (PyTorch, MediaPipe) | 2m   |
| 3    | Generate synthetic training data          | 10s  |
| 4    | Build Transformer model                   | 5s   |
| 5    | **Train the model** (main step)           | 1-2m |
| 6    | Evaluate and plot results                 | 30s  |
| 7    | Save checkpoint to Google Drive           | 10s  |

**Total time: ~4-6 minutes** (depending on GPU availability)

---

## Outputs You'll Get

After training, Google Drive will contain:

- **`tsl_transformer.pt`** (5-10 MB)
  - Your trained model checkpoint
  - Ready to use in the backend

- **`training_history.png`**
  - Loss and accuracy curves
  - Shows model learning progress

- **`confusion_matrix.png`**
  - Prediction accuracy per gesture
  - Helps identify weak gestures

- **`training_summary.txt`**
  - Complete training configuration
  - Model architecture details
  - Final metrics

---

## Advanced Options

### Use Different Training Parameters

Edit this cell before running training:

```python
EPOCHS = 50              # Number of training epochs
BATCH_SIZE = 32         # Batch size
LR = 1e-3              # Learning rate
SEQ_LENGTH = 20        # Sequence length
```

### Generate More Data

In the data generation step:

```python
X_train, y_train = generate_synthetic_dataset(
    num_samples=2000,   # Increase from 800
    seq_length=30       # Longer sequences
)
```

### Change Model Architecture

Modify the model initialization:

```python
model = TSLTransformer(
    d_model=256,              # Increase from 128
    num_encoder_layers=6,     # More layers
    nhead=8                   # More attention heads
)
```

---

## Troubleshooting

### "GPU not available"

- Colab GPU quota might be full
- Use CPU (slower) or try again later
- Or upgrade to Colab Pro

### "Google Drive mount failed"

- Make sure you're signed into Google
- Run Step 1 cell again

### "Module not found" error

- Run Step 2 (dependencies) again
- Clear cache: **Runtime** → **Restart runtime**

### "Checkpoint not saving"

- Check that `TSL-Checkpoints` folder exists in your Drive
- Run the checkpoint verification cell again

### Training is very slow

- Likely using CPU instead of GPU
- Check Step 1 output to confirm GPU
- Click **Runtime** → **Change runtime type** → **GPU**

---

## Local Training Alternative

Don't want to use Colab? Train locally instead:

```bash
cd backend
python train.py
```

This runs on your machine's GPU/CPU (much slower, but works with real data if you have it).

---

## Next Steps

1. ✅ Train model on Colab
2. ✅ Download checkpoint
3. ✅ Place in `backend/checkpoints/`
4. 🚀 Start the full TSL system with real gesture recognition!

---

## Resources

- **Colab Docs**: https://colab.research.google.com/notebooks/intro.ipynb
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **MediaPipe**: https://developers.google.com/mediapipe
- **Transformer Tutorial**: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

**Happy training! 🎉**
