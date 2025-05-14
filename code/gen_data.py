"midiファイルから数値データに変換するコード"
import mido
import numpy as np
import glob

# --- 定数設定 ---
MIDI_FILES = sorted(glob.glob('music/*.mid'))  # musicフォルダ内のMIDIファイル全取得
TIME_DIVISION = 16  # 16分音符
NOTE_MIN = 21  # ピアノ最低音 (A0)
NOTE_MAX = 108  # ピアノ最高音 (C8)
NUM_NOTES = NOTE_MAX - NOTE_MIN + 1  # ノート数

all_sequences = []  # 全MIDIデータのシーケンスを格納

for midi_file in MIDI_FILES:
    print(f"🎵 処理中: {midi_file}")
    
    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_sixteenth = ticks_per_beat // 4

    events = []
    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' or msg.type == 'note_off':
                events.append({'time': current_time, 'note': msg.note, 'velocity': msg.velocity, 'type': msg.type})

    if not events:
        continue  # ノート情報がないMIDIはスキップ

    total_ticks = events[-1]['time']
    total_steps = (total_ticks // ticks_per_sixteenth) + 1

    timeline = [set() for _ in range(total_steps)]
    active_notes = set()
    event_index = 0

    for step in range(total_steps):
        current_tick = step * ticks_per_sixteenth
        while event_index < len(events) and events[event_index]['time'] <= current_tick:
            event = events[event_index]
            note = event['note']
            if NOTE_MIN <= note <= NOTE_MAX:
                if event['type'] == 'note_on' and event['velocity'] > 0:
                    active_notes.add(note)
                else:
                    active_notes.discard(note)
            event_index += 1
        timeline[step] = active_notes.copy()

    one_hot_sequence = []
    for notes in timeline:
        one_hot = np.zeros(NUM_NOTES, dtype=np.int32)
        for note in notes:
            if NOTE_MIN <= note <= NOTE_MAX:
                one_hot[note - NOTE_MIN] = 1
        one_hot_sequence.append(one_hot)

    all_sequences.append(np.array(one_hot_sequence))

# --- 全データ結合 ---
final_dataset = np.concatenate(all_sequences, axis=0)
print(f"✅ 変換後データ形状: {final_dataset.shape}")  # (総時間ステップ数, ノート数)

# --- 保存 ---
np.save('data/all_sixteenth_note_dataset.npy', final_dataset)
print("✅ データセット保存完了: data/all_sixteenth_note_dataset.npy")
