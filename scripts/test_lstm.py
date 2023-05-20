import torch
from datasets.zebra_dataset_lstm import ZebraDatasetLSTM
import sys, os
from models.convlstm import ConvLSTM
import torch.optim as optim
import torch.nn as nn
from utils.utils import *


def test():
    # settings
    ckpt_dir = '../ckpt/epoch 11_val_loss_384.9290954871.pth'

    testset = ZebraDatasetLSTM('../data/val', 'val', window=11, size=128)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=0)

    model = ConvLSTM(
                 input_dim=3,
                 hidden_dim=[64, 64, 64, 64, 128, 128],
                 kernel_size=[(3, 3),(3, 3),(3, 3),(3, 3),(3, 3),(3, 3)],
                 final_conv_kernel_size=4,
                 num_layers=6,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
    model.load_state_dict(torch.load(ckpt_dir))
    model = model.cuda()

    model.eval()
    results = []
    gts_arr = []
    with torch.no_grad():
        for index, (inputs, gts) in enumerate(testloader):
            outputs = model(inputs)
            result = outputs.item()
            results.append(result)
            gts_arr.append(gts.item())
    print(results)
    print(gts_arr)






if __name__ == '__main__':
    test()