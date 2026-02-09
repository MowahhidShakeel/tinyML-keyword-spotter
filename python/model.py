import numpy as np
import pathlib
import tensorflow as tf

# Get all the commands (directory names)
data_dir = pathlib.Path("data/mini_speech_commands_extracted/mini_speech_commands")
commands = np.array(tf.io.gfile.listdir(str(data_dir)))

# Filter out README.md
commands = commands[commands != "README.md"]

print("Commands: ", commands)

# Define the target words
target_words = ["yes", "no"]

# Create file lists
filenames = tf.io.gfile.glob(str(data_dir) + "/*/*")
filenames = tf.random.shuffle(filenames)

print(f"Total audio files: {len(filenames)}")