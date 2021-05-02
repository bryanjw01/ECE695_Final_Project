from variables import NUM_WORKERS, BATCH_SIZE, NUM_CHANNELS, IMG_DIM, DATA_PATH
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import pdb

class DataFormatting:
    def __init__(self):
        '''
            data:
                Contains formatted data from the dataloader. The variable contents
                depend on the DATA_PATH from the variables file. If no data path
                specified then it defaults to using the MNIST dataset for training.
        '''
        if DATA_PATH:
            self.data = self.CreateDataset()
        else:
            self.data = self.CreateDatasetMNIST()


    def CreateDataset(self):
        dataset = torchvision.datasets.ImageFolder(
            root=DATA_PATH,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(IMG_DIM),
                torchvision.transforms.CenterCrop(IMG_DIM),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True, 
            num_workers=NUM_WORKERS
            )

        return dataloader
    

    def CreateDatasetMNIST(self):
        dataloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                '../../data/mnist',
                download=True,
                train=True,
                transform=torchvision.transforms.Compose
                ([
                    torchvision.transforms.Resize(IMG_DIM),
                    torchvision.transforms.CenterCrop(IMG_DIM),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5), (0.5))
                ])
            ),
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS
        )
        return dataloader
    

        

