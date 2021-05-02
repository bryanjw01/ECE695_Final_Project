import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from tfrecord_lite import decode_example
from torch.utils.data import DataLoader

from variables import TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TESTING_DATA_PATH, MOCHA_PATH, COLUMN_WIDTH, PHONEME_PATH
import pdb

class Data_Formatting:

    def __init__(self, EFC):
        self.vocab = self.read_vocab(MOCHA_PATH)
        self.vocab_phonemes = self.read_vocab(PHONEME_PATH)
        self.max_s_sz = None 
        self.max_ecog_sz, self.max_s_sz = self.max_data_size(TRAIN_DATA_PATH + VALIDATION_DATA_PATH + TESTING_DATA_PATH, COLUMN_WIDTH)
        self.train_data = self.data_format(TRAIN_DATA_PATH)
        self.val_data = self.data_format(VALIDATION_DATA_PATH)
        self.test_data = self.data_format(TESTING_DATA_PATH)

    

    def read_vocab(self, MOCHA_PATH)->list:
        
        vocab_dict = dict()
        
        with open(MOCHA_PATH, "r") as f:
            vocab_list = f.read().splitlines()
        
        vocab_dict = {k: v for v, k in enumerate(vocab_list)}
        
        return vocab_dict
    
    def phoneme_to_index(self, phoneme_sequence):
        for i in range(len(phoneme_sequence)):
            word = phoneme_sequence[i].decode('ascii') 
            if word in self.vocab_phonemes:
                phoneme_sequence[i] = self.vocab_phonemes[word]
            else:
                phoneme_sequence[i] = 1
  
        # phoneme_sequence =[0] + phoneme_sequence + [1] * (self.max_ecog_sz // COLUMN_WIDTH - len(phoneme_sequence) - 1)
        # phoneme_sequence = np.array(phoneme_sequence)[np.newaxis].T.tolist()
        phoneme_sequence = [0] + phoneme_sequence + [0] * (self.max_ecog_sz // COLUMN_WIDTH - len(phoneme_sequence) - 1)
        return phoneme_sequence[::12]


    def text_to_index(self, text_sequence):
        
        for i in range(len(text_sequence)):
            word = text_sequence[i].decode('ascii') 
            if word in self.vocab:
                text_sequence[i] = self.vocab[word]
            else:
                text_sequence[i] = 2
        
        if not self.max_s_sz:
            return [0] + text_sequence + [1]
        
        return [0] + text_sequence + [1] * (self.max_s_sz - len(text_sequence) - 1)
    

    def max_data_size(self, DATA_PATH, COLUMN_WIDTH):
        
        max_ecog_sz = 0
        max_s_sz = 0
        
        for path in DATA_PATH:
            it = tf.compat.v1.python_io.tf_record_iterator(path)
            for i in it:
                max_ecog_sz = max(len(decode_example(i)["ecog_sequence"]), max_ecog_sz)
                max_s_sz = max(len(self.text_to_index(decode_example(i)["text_sequence"])), max_s_sz)
        
        max_ecog_sz += COLUMN_WIDTH - (max_ecog_sz % COLUMN_WIDTH) 
        while (max_ecog_sz / COLUMN_WIDTH) % 12 != 0:
            max_ecog_sz += COLUMN_WIDTH
        return max_ecog_sz , max_s_sz


    def data_preprocessing(self, x):
        orig_shape = x.shape
        x.shape = (-1, COLUMN_WIDTH)
        scaler = preprocessing.StandardScaler()
        scaled_x = scaler.fit_transform(x)
        scaled_x.shape = orig_shape
        return scaled_x


    def data_format(self, DATA_PATH):
        # list of tokenized phoneme
        p = []
        # list of ecog data
        x = []
        # list of tokenized sentences
        y = []
        # list of sentences
        z = []

        for path in DATA_PATH:
            it = tf.compat.v1.python_io.tf_record_iterator(path)
            for i in it:
                ecog_sequence = decode_example(i)["ecog_sequence"]
                length = self.max_ecog_sz - len(ecog_sequence) 
                ecog_sequence = np.pad(ecog_sequence, (0, length), 'constant')
                ecog_sequence.shape = (-1, COLUMN_WIDTH)
                x.append(ecog_sequence)
                y.append(self.text_to_index(decode_example(i)["text_sequence"]))
                p.append(self.phoneme_to_index(decode_example(i)["phoneme_sequence"]))
                z.append(decode_example(i)["text_sequence"])
        x = self.data_preprocessing(np.array(x))
        print(len(p[-1]))
        df_data = pd.DataFrame({'Ecog Sequence': list(x), 'Text Sequence': y, 'Phoneme Sequence':p ,'Sentence': z})
        
        df_data = df_data[['Ecog Sequence','Text Sequence', 'Phoneme Sequence']].sample(len(df_data['Sentence']), random_state=999).to_numpy()
        
        data = DataLoader(df_data, collate_fn=lambda x: x, batch_size=500, pin_memory=True, shuffle=True)
        
        return data