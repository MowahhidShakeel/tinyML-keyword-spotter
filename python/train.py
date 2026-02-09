import tensorflow as tf
from tensorflow.keras import layers, models
import processing
import pathlib
import numpy as np

data_dir = pathlib.Path('data/mini_speech_commands_extracted/mini_speech_commands')

if not data_dir.exists():
    print(f"ERROR: Directory not found at {data_dir}")
    exit()

# Get all file paths
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')

# SAFETY CHECK: Stop if files not found
if len(filenames) == 0:
    print("ERROR: No files found! Check the path.")
    print(f"Looking in: {str(data_dir) + '/*/*'}")
    exit()
    
filenames = tf.random.shuffle(filenames)

# Split: 80% train, 10% validation, 10% test
num_samples = len(filenames)
train_files = filenames[:int(0.8*num_samples)]
val_files = filenames[int(0.8*num_samples): int(0.9*num_samples)]
test_files = filenames[int(0.9*num_samples):]

# Create the datasets
train_ds = processing.preprocess_dataset(train_files)
val_ds = processing.preprocess_dataset(val_files)
test_ds = processing.preprocess_dataset(test_files)

# Optimization: Batching and caching
BATCH_SIZE = 64
train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    # Input layer, matching the spectrogram shape
    layers.Input(shape=(124, 129, 1)),
    
    # Downscale to 32x32 to save RAM on microcontroller
    layers.Resizing(32,32),
    
    # Normalization, scale values from 0 to 1
    layers.Normalization(),
    
    # CNN Layers
    # First layer, to look for simple features
    layers.Conv2D(32, 3, activation='relu'),
    
    # Shrink the image by half through max pooling (model becomes 'invariant' to small shifts)
    layers.MaxPooling2D(),
    
    # Second layer, to look for complex features
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    
    # Randomly drop 25% of neurons to 0 during training
    # Forces the model to not rely on single neurons (prevents overfitting)
    layers.Dropout(0.25),
    
    # Unroll the 2D map into a 1D vector to feed the final layer
    layers.Flatten(),
    
    # Output layer with 3 neurons
    layers.Dense(3, activation='softmax')                       
])

# Print the model architecture
model.summary()

# Compile the model
model.compile(
    optimizer='adam',   # Standard smart optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train the model
EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save the model
model.save('saved_model.keras')