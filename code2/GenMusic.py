import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import keras
import mido
import pygame
from IPython import display
from matplotlib import pyplot as plt
from typing import Optional
import os

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000
sample_file = "music/praeludium1_1.mid"

pm = pretty_midi.PrettyMIDI(sample_file)

# midiファイルに使われている楽器を特定
def instruments():
    print('Number of instruments:', len(pm.instruments))
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    print('Instrument name:', instrument_name)
    
# 音楽の特徴量を表示    
def print_feature(instrument):    
    for i, note in enumerate(instrument.notes[:10]):
        note_name = pretty_midi.note_number_to_name(note.pitch)
        duration = note.end - note.start
        print(f'{i}: pitch={note.pitch}, note_name={note_name},'
                f' duration={duration:.4f}')    
     
# 音楽の特徴量をテーブルデータとして表示        
def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start
  
  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})        

# ピアノロールで表示
def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)
  plt.show()
  
# 各特徴量の分布を表示  
def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
  plt.figure(figsize=[15, 5])
  plt.subplot(1, 3, 1)
  sns.histplot(notes, x="pitch", bins=20)

  plt.subplot(1, 3, 2)
  max_step = np.percentile(notes['step'], 100 - drop_percentile)
  sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))
  
  plt.subplot(1, 3, 3)
  max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
  sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))  
  
# 音楽の特徴量をMIDIファイルに変換  
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def save_dataset(dataset, file_path):
    # シリアライズ関数
    def serialize_example(inputs, labels):
        feature = {
            'inputs': tf.train.Feature(float_list=tf.train.FloatList(value=inputs.numpy().flatten())),
            'labels_pitch': tf.train.Feature(float_list=tf.train.FloatList(value=labels['pitch'].numpy().flatten())),
            'labels_step': tf.train.Feature(float_list=tf.train.FloatList(value=labels['step'].numpy().flatten())),
            'labels_duration': tf.train.Feature(float_list=tf.train.FloatList(value=labels['duration'].numpy().flatten())),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


    with tf.io.TFRecordWriter(file_path) as writer:
        for inputs, labels in dataset:
            serialized_example = serialize_example(inputs, labels)
            writer.write(serialized_example)


# データセット生成関数
def prepare_dataset(filenames, num_files=5, seq_length=25, vocab_size=128, batch_size=64):
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)  # MIDI → テーブルデータ
        all_notes.append(notes)
    all_notes = pd.concat(all_notes)
    
    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
    
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    
    # --- シーケンス化 ---
    def create_sequences(dataset, seq_length, vocab_size):
        seq_length += 1  # ラベル用に1増やす

        windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)
        sequences = windows.flat_map(lambda x: x.batch(seq_length, drop_remainder=True))

        # 正規化: pitch だけ正規化
        def scale_pitch(x):
            return x / [vocab_size, 1.0, 1.0]

        # 入力とラベルに分割
        def split_labels(seq):
            inputs = seq[:-1]
            labels_dense = seq[-1]
            labels = {key: labels_dense[i] for i, key in enumerate(key_order)}
            return scale_pitch(inputs), labels

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    
    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)

    # --- バッチ処理 ---
    n_notes = len(train_notes)
    buffer_size = n_notes - seq_length
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.AUTOTUNE))
    
    save_dataset(train_ds, 'dataset/train_ds.tfrecord')


class LSTMModel(keras.Model):
    def __init__(self, lstm_units=128, output_dim=3):
        super(LSTMModel, self).__init__()
        
        # LSTM層
        self.lstm1 = keras.layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = keras.layers.LSTM(lstm_units)
        
        # 出力層
        self.dense = keras.layers.Dense(output_dim)
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        return self.dense(x)
      
    
def parse_example(example_proto):
    seq_length=25
    feature_description = {
        'inputs': tf.io.FixedLenFeature([seq_length * 3], tf.float32),  # (シーケンス長 * 特徴量数)
        'labels_pitch': tf.io.FixedLenFeature([], tf.float32),
        'labels_step': tf.io.FixedLenFeature([], tf.float32),
        'labels_duration': tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    inputs = tf.reshape(example['inputs'], (seq_length, 3))
    labels = {'pitch': example['labels_pitch'],
              'step': example['labels_step'],
              'duration': example['labels_duration']}
    return inputs, labels

def load_tfrecord_dataset(tfrecord_path, batch_size, parse_fn):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = raw_dataset.map(parse_fn)
    dataset = dataset.batch(batch_size)
    return dataset  

def load_dataset(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    return raw_dataset.map(parse_example)


def train_model(model, train_ds, val_ds=None, epochs=50, checkpoint_dir='./training_checkpoints', patience=5):

    # チェックポイントディレクトリの作成
    os.makedirs(checkpoint_dir, exist_ok=True)

    # コールバックの設定
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_{epoch}.weights.h5')
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        )
    ]

    # モデルのコンパイル
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=['mae']
    )

    # モデルの学習
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 学習済みモデルの保存
    model.save('trained_model.keras')

    return history
    
    
if __name__ == "__main__":
    select = ["Gen_data", "Train"]
    choice = input(f"Choose a method to execute ({', '.join(select)}): ")
    filenames = glob.glob('music/*.mid')
    model = LSTMModel(lstm_units=128, output_dim=3)
    train_ds = load_tfrecord_dataset(
    tfrecord_path='dataset/train_ds.tfrecord',
    batch_size=64,
    parse_fn=parse_example
)

    if choice == "Gen_data":
        train_ds = prepare_dataset(filenames, num_files=5, seq_length=25, vocab_size=128, batch_size=64)
    elif choice == "Train":
        train_model(model, train_ds, val_ds=None, epochs=50, checkpoint_dir='./training_checkpoints', patience=5)
    else:
        print("❌ 無効な選択肢です。")

  