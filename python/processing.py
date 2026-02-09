import tensorflow as tf
import os 

# Constants
SAMPLE_RATE = 16000    # 16kHz is the standard for voice
FRAME_LENGTH = 255      # 255 samples = ~16ms (Enough to capture a phoneme)
FRAME_STEP = 128        # 128 samples = ~8ms (50% overlap for smooth transitions)

def get_label(file_path):
    # Extract the folder name from the path (e.g., "yes" from ".../yes/file.wav")
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
  
    # If the label is not 'yes' or 'no', return 'unknown'
    return label


def get_label_id(file_path):
    # Splits the path strings by '\' or '/'
    parts = tf.strings.split(file_path, os.path.sep)
    # Grab the folder name (e.g. "yes")
    label = parts[-2]
    
    is_yes = label == 'yes'
    is_no = label == 'no'
    
    # Convert True/False to 1/0
    # if yes: is_yes=1, is_no=0
    # if no:  is_yes=0, is_no=1
    # if unk: is_yes=0, is_no=0
    is_yes_int = tf.cast(is_yes, tf.int32)
    is_no_int = tf.cast(is_no, tf.int32)
    
    # Calculate ID, Target: Yes=0, No=1, Unknown=2
    # Formula: 2 - (2 * yes) - (1 * no)
    # Ex: Yes (1,0) -> 2 - 2 - 0 = 0
    # Ex: No  (0,1) -> 2 - 0 - 1 = 1
    # Ex: Unk (0,0) -> 2 - 0 - 0 = 2
    return 2 - (2 * is_yes_int) - is_no_int


def get_spectrogram(audio):
    # Padding, ensure audio is exactly 16000 samples
    # If shorter, pad with 0s. If longer, crop
    audio = audio[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    
    # STFT, convert time -> frequency
    spectrogram = tf.signal.stft(
        audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP
    )
    
    # Get the magnitude only (loudness) of STFT return (complex number a+ib)
    spectrogram = tf.abs(spectrogram)
    
    # Adding channel dimensions since CNNs expect inputs like h, w, channels
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    
    return spectrogram


def get_spectrogram_from_path(file_path):
    # Helper to combine audio decoding + spectrogram generation
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)
    
    return get_spectrogram(audio)


def preprocess_dataset(file_paths):
    # Create dataset from the filenames
    files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    
    # Map the 'loader' function to every file
    output_ds = files_ds.map(lambda x: (get_spectrogram_from_path(x), get_label_id(x)),
                             num_parallel_calls=tf.data.AUTOTUNE)
    
    return output_ds
    