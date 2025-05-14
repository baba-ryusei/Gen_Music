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

# --- AutoEncoder定義 ---
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(88, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 88),
            nn.Sigmoid(),  # 出力を0〜1に
        )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def train():
    model = AutoEncoder().to(device)  # モデルをデバイスに移動
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 100

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            x_batch = batch[0].to(device)  # バッチもデバイスに移動
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, x_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'model/autoencoder_music.pth')
    print("✅ モデル保存完了")

def gen():
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load('model/autoencoder_music.pth', map_location=device))
    model.eval()

    latent_dim = 16
    generated_sequence = []

    for _ in range(576):
        z = torch.randn(1, latent_dim).to(device)  # 潜在ベクトルもデバイス上で生成
        with torch.no_grad():
            generated = model.decoder(z)
        one_hot = (generated > 0.5).int().cpu().numpy().squeeze()  # CPUに戻す
        generated_sequence.append(one_hot)

    generated_sequence = np.array(generated_sequence)
    print(generated_sequence.shape)

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = 480
    sixteenth_note_ticks = ticks_per_beat // 4

    for one_hot in generated_sequence:
        notes = np.where(one_hot == 1)[0] + NOTE_MIN
        for note in notes:
            track.append(Message('note_on', note=int(note), velocity=64, time=0))
        track.append(Message('note_off', note=0, velocity=0, time=sixteenth_note_ticks))

    mid.save('generated_music.mid')
    print("✅ 生成MIDI保存完了: generated_music.mid")


if __name__ == '__main__':
    
    select = ["Train", "Gen"]
    choice = input(f"Choose a method to execute ({', '.join(select)}): ")

    if choice == "Train":
        train()
    elif choice == "Gen":
        gen()
    else:
        print("❌ 無効な選択肢です。")


    
    
