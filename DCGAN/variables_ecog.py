import torch

VALIDATION_DATA_PATH = ["/content/drive/My Drive/tf_records/EFC401_B83.tfrecord"]
TRAIN_DATA_PATH = [f"/content/drive/My Drive/tf_records/EFC401_B{i}.tfrecord" for i in [4, 41, 57,61, 66, 73, 77, 87]]
TESTING_DATA_PATH = ["/content/drive/My Drive/tf_records/EFC401_B69.tfrecord"]
COLUMN_WIDTH = 429

MOCHA_PATH = "/content/drive/My Drive/tf_records/vocab.mocha-timit.1806"
PHONEME_PATH = "/content/drive/MyDrive/tf_records/vocab.phonemes.42"

ENCODER_INPUT = 0
DECODER_INPUT = 1 
PHONEME_INPUT = 2

#  Model parameters
INPUT_DIM = 5000
OUTPUT_DIM = 5000
D_MODEL = 256
N = 2
H = 8
D_FF = 512
DROPOUT = 0.1
LEARNING_RATE = 0.0001
TRG_PAD_IDX = 1
SRC_PAD_IDX = 1

N_EPOCHS = 250
CLIP = 1
N_VAL = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')