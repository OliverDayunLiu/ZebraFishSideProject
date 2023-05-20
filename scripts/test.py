import torch
from datasets.zebra_dataset import ZebraDataset
import sys, os
from torchvision import models
import torch.nn as nn
from utils.utils import *
import matplotlib.pyplot as plt
import numpy as np

def train_and_val():
    # settings
    save_dir = '../ckpt'
    checkpoint = os.path.join(save_dir, 'epoch 15_val_loss_0.1283639826.pth')

    testset = ZebraDataset('../data/test_control35', 'val', size=256)
    #testset = ZebraDataset('../data/test_methylone5', 'val', size=256)
    #testset = ZebraDataset('../data/test_dimethylone7', 'val', size=256)
    #testset = ZebraDataset('../data/test_MDMA8', 'val', size=256)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    #model = models.vgg16_bn(pretrained=True)
    #model.classifier[-1] = nn.Linear(in_features=4096, out_features=1024)
    #model.classifier.add_module('relu', nn.ReLU())
    #model.classifier.add_module('last_linear', nn.Linear(in_features=1024, out_features=1))

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 1)

    model.load_state_dict(torch.load(checkpoint))
    model = model.cuda()

    model.eval()
    results = []
    gts_arr = []
    error = []
    with torch.no_grad():
        for index, (inputs, gts) in enumerate(testloader):
            outputs = model(inputs)
            result = outputs.item()
            results.append(result)
            gts_arr.append(gts.item())
            error.append(abs(gts.item()-result))

    x = np.arange(1, len(results)+1)
    plt.plot(x, gts_arr, label='gts', color='red')

    results = np.array(results)

    plt.plot(x, results, label='Atrium prediction', color='green')
    plt.legend()
    plt.show()

    print("error: ", np.sum(error)/len(testloader))

if __name__ == '__main__':
    train_and_val()