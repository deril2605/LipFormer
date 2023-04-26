import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, MaxPool3D, Conv3D, TimeDistributed, Flatten, Dropout, Bidirectional

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_video(path):
    """
    Helper function to load a video sequentially as frames and standardize them
    Each frame is cropped to capture around the lip regions to reduce computations
    and improve accuracy
    """
    frames = []
    cap = cv2.VideoCapture(path)
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        _, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:]) # hardcoding the lip coordinates
    cap.release()
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def define_vocab_mapping(vocab):
    """
    Helper function to define simple vocabulary mapping for the unique characters in video sequences.
    """
    character_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_character = tf.keras.layers.StringLookup(vocabulary=character_to_num.get_vocabulary(), oov_token="", invert=True)
    print(
    f"The vocabulary is: {character_to_num.get_vocabulary()} "
    f"(size ={character_to_num.vocabulary_size()})"
    )
    return character_to_num, num_to_character

def load_align(path, char_to_num):
    """
    Helper function to load an alignment and map it with the vocabulary defined
    """
    with open(path, 'r') as f: 
        lines = f.readlines()
    tokens = []
    for line in lines:
        words = line.split(' ')
        if words[2] != 'sil': # ignore timestamps that are marked silence
            tokens = [*tokens,' ',words[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path):
    """
    Helper function to load frames of a video and corresponding alignments for the path provided.
    """
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    align_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_align(align_path, char_to_num)
    return frames, alignments[0:40] # trim alignments to 40 window

def mappable_function(path):
    """
    Helper function to wrap `load_data` to a tensorFlow op that executes it eagerly.
    """
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

def get_model(char_to_num, load_weights = False):
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same', activation = 'relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same', activation = 'relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same', activation = 'relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))
    if load_weights:
        model.load_weights('models/checkpoint')
    return model

def CTCLoss(y_true, y_pred):
    """
    Define custom CTC Loss function
    https://keras.io/examples/audio/ctc_asr/#model
    CTC is an algorithm used to train deep neural networks in speech recognition, 
    handwriting recognition and other sequence problems. CTC is used when we don’t know how 
    the input aligns with the output (how the characters in the transcript align to the audio). 
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "] # all set of characters in dataset
char_to_num, num_to_char = define_vocab_mapping(vocab)