import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mido import Message, MidiFile, MidiTrack

NOTE_MIN = 21   # ピアノ最低音 (A0)
X = np.load('data/sixteenth_note_dataset.npy')

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# --- データセット作成 ---
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # データもデバイスに移動
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for i in range(0,576):
    print(X_tensor[i])