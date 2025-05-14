import mido
import glob

MIDI_FILES = sorted(glob.glob('music/*.mid'))

for midi_file in MIDI_FILES:
    try:
        mid = mido.MidiFile(midi_file)
        print(f"✅ 読み込み成功: {midi_file}")
    except Exception as e:
        print(f"❌ 読み込み失敗: {midi_file} → {e}")
