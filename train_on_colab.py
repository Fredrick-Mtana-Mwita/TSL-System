"""
Google Colab Training Script for TSL Transformer

Instructions:
1. Upload this file to Google Colab
2. Run all cells
3. The trained model will be saved to your Google Drive at: /MyDrive/TSL-Checkpoint/tsl_transformer.pt
4. Download it and place in backend/checkpoints/ folder locally

OR use the notebook link: https://colab.research.google.com/
"""

# =============================================================================
# CELL 1: Mount Google Drive and Install Dependencies
# =============================================================================

# Mount Google Drive
try:
    from google.colab import drive  # type: ignore[import-untyped]
    drive.mount('/content/drive')
except ImportError:
    print("Not running in Colab — skipping Drive mount.")

# Install required packages
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "torch>=2.2.0",
                      "mediapipe>=0.10.0",
                      "numpy",
                      "scikit-learn",
                      "tqdm"])

print("✅ Dependencies installed successfully!")

# =============================================================================
# CELL 2: Download and Extract Project from GitHub (OR Upload ZIP)
# =============================================================================

import os
import shutil

# Option A: Clone from GitHub (if you have a repo)
# Uncomment and modify if using GitHub
# !git clone https://github.com/YOUR_USERNAME/TSL-System.git /content/TSL-System

# Option B: Use uploaded files
# Assume you uploaded the project folder to /content/TSL-System
# Or create it from your local files

# For this example, let's create the directory structure
project_dir = "/content/TSL-System"
os.makedirs(project_dir, exist_ok=True)
os.makedirs(f"{project_dir}/backend", exist_ok=True)
os.makedirs(f"{project_dir}/backend/checkpoints", exist_ok=True)

print(f"✅ Project directory created at {project_dir}")

# =============================================================================
# CELL 3: Copy Project Files
# =============================================================================

# You can either:
# 1. Upload a ZIP and extract it
# 2. Copy files manually from your local machine
# 3. Clone from a GitHub repository

# Example: If you uploaded files to /content/files/
# shutil.copytree("/content/files/backend", f"{project_dir}/backend", dirs_exist_ok=True)

# For now, we'll assume you'll upload the backend folder
print("📁 Project structure:")
print(os.listdir(project_dir))

# =============================================================================
# CELL 4: Import and Setup (Create inline models if needed)
# =============================================================================

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, f"{project_dir}/backend")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Using device: {DEVICE}")

# =============================================================================
# CELL 5: Define Transformer Model (Inline)
# =============================================================================

NUM_CLASSES = 12

class TSLTransformer(nn.Module):
    """Transformer model for sign language gesture classification."""
    
    def __init__(self, num_classes=NUM_CLASSES, d_model=128, nhead=4, 
                 num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection: 126 features (2 hands × 21 landmarks × 3 coords)
        self.input_projection = nn.Linear(126, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 20, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, 126) - hand landmark sequences
        """
        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer encoding
        x = self.encoder(x)  # (batch, seq, d_model)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification
        logits = self.fc(x)  # (batch, num_classes)
        return logits

# =============================================================================
# CELL 6: Define Synthetic Dataset Generator
# =============================================================================

def generate_synthetic_dataset(num_samples=1000, seq_length=20):
    """Generate synthetic hand landmark data for testing."""
    print(f"Generating {num_samples} synthetic samples...")
    
    X = []
    y = []
    
    for i in range(num_samples):
        # Generate random hand landmarks (21 landmarks × 3 coords × 2 hands = 126)
        sequence = np.random.randn(seq_length, 126).astype(np.float32)
        
        # Add some structure per class
        class_label = np.random.randint(0, NUM_CLASSES)
        sequence[:, :10] *= (class_label + 1) * 0.1  # Simple class-dependent variation
        
        X.append(sequence)
        y.append(class_label)
    
    return np.array(X), np.array(y)

class TSLDataset(Dataset):
    """PyTorch Dataset for hand landmarks."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# =============================================================================
# CELL 7: Training Loop
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device: "torch.device | str" = "cpu"):
    """Train the Transformer model."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            checkpoint_path = f"{project_dir}/backend/checkpoints/tsl_transformer.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
                break
    
    return model

# =============================================================================
# CELL 8: Generate Data and Train
# =============================================================================

print("=" * 60)
print("🚀 TSL Transformer Training on Google Colab")
print("=" * 60)

# Generate synthetic data
X_train, y_train = generate_synthetic_dataset(num_samples=800, seq_length=20)
X_val, y_val = generate_synthetic_dataset(num_samples=200, seq_length=20)

print(f"✅ Generated training data: {X_train.shape}")
print(f"✅ Generated validation data: {X_val.shape}")

# Create datasets and dataloaders
train_dataset = TSLDataset(X_train, y_train)
val_dataset = TSLDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = TSLTransformer(
    num_classes=NUM_CLASSES,
    d_model=128,
    nhead=4,
    num_encoder_layers=4,
    dim_feedforward=256
)

print(f"\n📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
print("\n⏳ Starting training...")
train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device=DEVICE)

print("\n✅ Training complete!")

# =============================================================================
# CELL 9: Save to Google Drive
# =============================================================================

checkpoint_src = f"{project_dir}/backend/checkpoints/tsl_transformer.pt"
drive_dir = "/content/drive/MyDrive/TSL-Checkpoint"
os.makedirs(drive_dir, exist_ok=True)
checkpoint_dst = f"{drive_dir}/tsl_transformer.pt"

shutil.copy(checkpoint_src, checkpoint_dst)
print(f"✅ Checkpoint saved to Google Drive: {checkpoint_dst}")
print(f"\n📥 Download it from your Google Drive: https://drive.google.com/drive/my-drive")

# =============================================================================
# CELL 10: Download Link
# =============================================================================

try:
    from google.colab import files as _colab_files  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    pass

print("\n" + "=" * 60)
print("📥 DOWNLOAD INSTRUCTIONS")
print("=" * 60)
print(f"✅ Model checkpoint saved to: /TSL-Checkpoint/tsl_transformer.pt")
print(f"\nTo use this model locally:")
print(f"1. Create folder: backend/checkpoints/")
print(f"2. Move the downloaded file to: backend/checkpoints/tsl_transformer.pt")
print(f"3. Restart the backend server")
print(f"\nYour trained model is ready! 🎉")
