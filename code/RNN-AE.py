import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mido import Message, MidiFile, MidiTrack

SEQUENCE_LENGTH = 512  # 好きな長さに設定

X = np.load('data/sixteenth_note_dataset.npy')  # (576, 88)
print(f"元データ形状: {X[20:30]}")

X_sequences = []
for i in range(0, len(X) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
    X_sequences.append(X[i:i+SEQUENCE_LENGTH])

X_sequences = np.array(X_sequences)
print(f"変換後データ形状: {X_sequences.shape}")  # (サンプル数, シーケンス長, 88)


NOTE_MIN = 21   # ピアノ最低音 (A0)

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- データセット作成 ---
X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class RNNAutoEncoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=128, latent_dim=64, num_layers=2):
        super(RNNAutoEncoder, self).__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        self.decoder_output = nn.Sequential(
    nn.Linear(hidden_dim, input_dim),
    nn.Sigmoid()
)
    
    def forward(self, x):
        # --- Encoder ---
        print("input=",x)
        _, (h_n, _) = self.encoder_lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        h_last = h_n[-1]  # 最後の層の隠れ状態
        z = self.encoder_fc(h_last)  # 潜在ベクトル: (batch, latent_dim)
        
        # --- Decoder ---
        h_dec = self.decoder_fc(z).unsqueeze(0).repeat(self.decoder_lstm.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        c_dec = torch.zeros_like(h_dec)  # LSTMのcell state
        
        # デコーダー入力（全部0でもOK、今回は最初の入力として）
        decoder_input = torch.zeros_like(x)
        out, _ = self.decoder_lstm(decoder_input, (h_dec, c_dec))
        out = self.decoder_output(out)
        print("output=",out)
        
        return out

def min_max_normalize(x):
    min_val = x.min(dim=0, keepdim=True).values
    max_val = x.max(dim=0, keepdim=True).values
    return (x - min_val) / (max_val - min_val + 1e-8)  # 分母が0にならないよう微小値加算


def train():
    model = RNNAutoEncoder().to(device)
    criterion = nn.BCELoss()  # 出力が0~1なのでBCELoss
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    EPOCHS = 1000

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            x_batch = batch[0].to(device) #(batchsize * 88)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, x_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), 'model/rnn-autoencoder_music.pth')
    print("✅ モデル保存完了")

def gen():
    model = RNNAutoEncoder().to(device)
    model.load_state_dict(torch.load('model/rnn-autoencoder_music.pth', map_location=device))
    model.eval()

    latent_dim = 64  # モデルのlatent_dimに合わせる！
    sequence_length = 576

    # --- 潜在ベクトル z を生成 ---
    z = torch.rand(1, latent_dim).to(device)
    #z = torch.relu(z)
    print("z=",z)

    # --- デコーダーからシーケンス全体を生成 ---
    with torch.no_grad():
        h_dec = model.decoder_fc(z).unsqueeze(0).repeat(model.decoder_lstm.num_layers, 1, 1)
        c_dec = torch.zeros_like(h_dec)
        
        decoder_input = torch.zeros(1, sequence_length, 88).to(device)  # 空の入力
        generated = model.decoder_lstm(decoder_input, (h_dec, c_dec))[0]
        generated = model.decoder_output(generated)  # (1, 576, 88)

    # --- ワンホット化 ---
    generated_sequence = (generated > 0.5).int().cpu().numpy().squeeze()  # (576, 88)
    print("rithm",generated_sequence[20:30])

    # --- MIDIファイルに変換 ---
    from mido import Message, MidiFile, MidiTrack

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
       
        
def gen_with_input():
    model = RNNAutoEncoder().to(device)
    model.load_state_dict(torch.load('model/rnn-autoencoder_music.pth', map_location=device))
    model.eval()

    latent_dim = 64  # モデルのlatent_dimに合わせる！
    sequence_length = SEQUENCE_LENGTH  # 16

    # --- 入力データを読み込み ---
    #X = np.load('data/sixteenth_note_dataset.npy')
    X = np.load('data/all_sixteenth_note_dataset.npy')
    X_tensor = torch.tensor(X[:sequence_length], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 16, 88)

    # --- エンコーダーで潜在ベクトルを取得 ---
    with torch.no_grad():
        _, (h_n, _) = model.encoder_lstm(X_tensor)
        h_last = h_n[-1]
        z_0 = model.encoder_fc(h_last)  # 潜在ベクトル z₀

    # --- ノイズを加算 ---
    noise = torch.randn_like(z_0) * 0.3  # ノイズ強さ調整
    z_new = z_0 + noise

    # --- デコーダーで音楽生成 ---
    with torch.no_grad():
        h_dec = model.decoder_fc(z_new).unsqueeze(0).repeat(model.decoder_lstm.num_layers, 1, 1)
        c_dec = torch.zeros_like(h_dec)
        
        decoder_input = torch.zeros(1, sequence_length, 88).to(device)  # 空の入力
        generated = model.decoder_lstm(decoder_input, (h_dec, c_dec))[0]
        generated = model.decoder_output(generated)  # (1, 16, 88)

    # --- ワンホット化 ---
    print("generated_size=",generated.size())
    print("generated=",generated)
    generated_sequence = (generated*(10**5) > 0.5).int().cpu().numpy().squeeze()  # (16, 88)

    # --- MIDI変換 ---
    from mido import Message, MidiFile, MidiTrack

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = 480
    sixteenth_note_ticks = ticks_per_beat // 4

    for one_hot in generated_sequence:
        notes = np.where(one_hot == 1)[0] + NOTE_MIN
        for note in notes:
            duration_options = [sixteenth_note_ticks, sixteenth_note_ticks * 2, sixteenth_note_ticks * 3, sixteenth_note_ticks * 4]  # 16分, 8分, 付点8分, 4分
            duration = np.random.choice(duration_options)
            #track.append(Message('note_off', note=0, velocity=0, time=duration))
            track.append(Message('note_on', note=int(note), velocity=64, time=duration))
        track.append(Message('note_off', note=0, velocity=0, time=duration))
        #track.append(Message('note_off', note=0, velocity=0, time=sixteenth_note_ticks))

    mid.save('generated_music_with_input.mid')
    print("✅ 入力から変化を加えて生成: generated_music_with_input.mid")
    
def gen_autoregressive():
    model = RNNAutoEncoder().to(device)
    model.load_state_dict(torch.load('model/rnn-autoencoder_music.pth', map_location=device))
    model.eval()

    latent_dim = 64
    sequence_length = 576
    input_dim = 88

    # 潜在ベクトルを生成
    z = torch.rand(1, latent_dim).to(device)

    with torch.no_grad():
        # 初期状態（h, c）を潜在ベクトルから作成
        h_dec = model.decoder_fc(z).unsqueeze(0).repeat(model.decoder_lstm.num_layers, 1, 1)
        c_dec = torch.zeros_like(h_dec)

        input_t = torch.zeros(1, 1, input_dim).to(device)  # 最初の1ステップだけ0ベクトル

        outputs = []

        for _ in range(sequence_length):
            out_t, (h_dec, c_dec) = model.decoder_lstm(input_t, (h_dec, c_dec))  # 1ステップ生成
            output_t = model.decoder_output(out_t)  # (1, 1, 88)
            outputs.append(output_t)
            input_t = output_t  # 次の入力に

        generated = torch.cat(outputs, dim=1)  # (1, 576, 88)

    # --- ワンホット化 ---
    #generated = min_max_normalize(generated)
    generated = torch.nn.functional.normalize(generated)
    print("generated",generated)
    #generated_sequence = generated.int().cpu().numpy().squeeze()
    generated_sequence = (generated > 0.5).int().cpu().numpy().squeeze()
    print("rithm",generated_sequence[:10])

    # --- MIDI変換はそのままでOK ---
    # --- MIDIファイルに変換 ---
    from mido import Message, MidiFile, MidiTrack

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
    
    select = ["Train", "Gen_noise", "Gen_input", "Gen_auto"]
    choice = input(f"Choose a method to execute ({', '.join(select)}): ")

    if choice == "Train":
        train()
    elif choice == "Gen_noise":
        gen()
    elif choice == "Gen_input":
        gen_with_input()    
    elif choice == "Gen_auto":
        gen_autoregressive()    
    else:
        print("❌ 無効な選択肢です。")


    
    