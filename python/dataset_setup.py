import os
import pathlib
import tensorflow as tf

# Use the 'mini' dataset for quick testing, or the full one for production.
DATASET_URL = 'http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip'
data_dir = pathlib.Path('data/mini_speech_commands')

if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin=DATASET_URL,
      extract=True,
      cache_dir='.',      # Save in current directory
      cache_subdir='data' # Inside a 'data' folder
  )

print(f"Data downloaded to: {data_dir}")