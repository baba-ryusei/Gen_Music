import mido
import torch
import torch.nn.functional as F

midi_file1 = "music/praeludium1_1.mid"
#midi_file = "generated_music.mid"
midi_file2 = "generated_music_with_input.mid"


def note(midi_file):
    mid = mido.MidiFile(midi_file)
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" or msg.type == "note_off":
                print(msg.note)
                onehot = F.one_hot(torch.tensor(msg.note),num_classes=127)
                print(onehot)
                
def time(midi_file):
    mid = mido.MidiFile(midi_file)
    for track in mid.tracks:
        for msg in track:
            print(msg)
           
# PPQ=480 ⇨ 16分音符は120tick                    
def tick(midi_file):
    mid = mido.MidiFile(midi_file)
    print(mid.ticks_per_beat)
      
#note()
time(midi_file1)                 
#time(midi_file2)                
            