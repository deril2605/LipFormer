import os
import cv2
import sys
import warnings
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
import gdown
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers.legacy import Adam
from utils import define_vocab_mapping, load_data, mappable_function, get_model, CTCLoss

warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "] # all set of characters in dataset
char_to_num, num_to_char = define_vocab_mapping(vocab)

class ProduceExample(tf.keras.callbacks.Callback):
    """
    Custom callback to print original and predicted alignment after epoch end
    """
    def __init__(self, dataset) -> None: 
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)

def download_data():
    if not os.path.exists('data.zip'):
        url = 'https://drive.google.com/uc?id=1m5J8VO-7uK3BmaYEh1nX2JBHP35axsh-'
        output = 'data.zip'
        gdown.download(url, output, quiet=False,use_cookies=False)
        gdown.extractall('data.zip')
    return 0

def visualize_data(path, num_to_char):
    frames, alignments = load_data(tf.convert_to_tensor(path))
    plt.imshow(frames[45])
    plt.savefig('resources/sample_frame.png', dpi=100)

    print(tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()]))

def train():
    download_data()

    visualize_data('data/s1/pwad3s.mpg', num_to_char)

    # load all files from folder with .mpg extension
    data = tf.data.Dataset.list_files('data/s1/*.mpg')

    # shuffle data
    data = data.shuffle(500, reshuffle_each_iteration=False)

    # map load_data across the elements of dataset
    data = data.map(mappable_function)

    # pad data frames to 75 and alignments to 40 and create batches with 2 elements
    data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))

    # improve latency by prefetched data
    data = data.prefetch(tf.data.AUTOTUNE)

    # split into train test batches 900-100 split
    test = data.enumerate() \
                        .filter(lambda x,y: x % 10 == 0) \
                        .map(lambda x,y: y)

    train = data.enumerate() \
                        .filter(lambda x,y: x % 10 != 0) \
                        .map(lambda x,y: y)

    # print shape of data
    print(data.as_numpy_iterator().next()[0][0][0].shape)
    # (75, 46, 140, 1)
    # 75 frames, 46 height of each frame, 140 width of each frame, 1 color channel

    model = get_model(char_to_num)
    print(model.summary())

    def scheduler(epoch, lr):
        """
        Dynamic scheduler to improve training time
        """
        if epoch < 30:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    # Define optimizer and callbacks
    lr = 0.0001
    model.compile(optimizer=Adam(learning_rate=lr), loss=CTCLoss)
    checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint'), monitor='loss', save_weights_only=True)
    schedule_callback = LearningRateScheduler(scheduler)
    example_callback = ProduceExample(test)

    model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback, example_callback])

def predict():
    model = get_model(char_to_num, load_weights = True)

    sample = load_data(tf.convert_to_tensor('test_data/pgwr5s.mpg'))

    real = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in [sample[1]]]
    real = real[0].numpy().decode('utf-8').strip('sil').strip()
    
    yhat = model.predict(tf.expand_dims(sample[0], axis=0))
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    pred = [tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in decoded]
    print('~'*100, 'REAL TEXT')
    print(real)
    print('~'*100, 'PREDICTIONS')
    print(pred[0])
def main(argv):

    # configure GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    type = "predict" # set the mode to either ["train", "predict"]

    if type == "train":
        train()

    elif type == "predict":
        predict()

    return

if __name__ == "__main__":
    main(sys.argv)