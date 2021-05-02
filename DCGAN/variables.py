import torch

'''
    NUM_WORKERS:
        variable controls how the dataset and more especifically, the batches, 
        are loaded into the gpu. This is needed due to the size of the data as well as
        it helps with efficiency.

    BATCH_SIZE:
        Specifies the batch size for training and testing dataset. We need to keep it 
        small to reduce training time as well as reducing the chances of model failing
        due to instability.

    NUM_CHANNELS:
        Specifies the number of channels that an image has. An rgb image has 3 channels
        while an greyscale image only has 1 channel. Image -> (channel, dim, dim)

    IMG_DIM:
        This specifies the dimensions of the images (height and width). In this experiment,
        we are using this variable to reshape the image into a square image therefore both
        height and width will be equivalent to each other. 
        Image -> (NUM_CHANNELS, IMG_DIM, IMG_DIM) after reshaping.

    NUM_EPOCHS: 
        Especifies the number of epochs needed to train the model.

    LEARNING_RATE:
        Initialized at 0.0001, but since we are using adam optimizer, adam will
        adjust it for us based on the model needs.

    DATA_PATH:
        Especifies the data path for the training and testing data. If path is
        empty then system defaults to using the MNIST dataset for data generation.


'''
DATA_PATH = ''

NUM_WORKERS = 2

BATCH_SIZE = 64

NUM_CHANNELS = 1
if DATA_PATH:
    NUM_CHANNELS = 3

IMG_DIM = 64

NUM_EPOCHS = 5

LEARNING_RATE = 0.0002

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


