import numpy as np
import torchvision
import matplotlib.pyplot as plt
from variables import DEVICE

class VerifyDataloader:
    
    def __init__(self, data):
        self.data = data
    
    def VerifyData(self, title="Training Images"):
        '''
            VerifyData:
                plots some of the training images
        '''
        real_batch = next(iter(self.data))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(title)
        plt.imshow(
            np.transpose(
                torchvision.utils.make_grid
                (
                    real_batch[0].to(DEVICE)[:64], 
                    padding=2, 
                    normalize=True
                ).cpu(),
                (1,2,0)
            )
        )

class ViewData:
    pass