import torch
from datasets.zebra_dataset import ZebraDataset
import sys, os
from models.convlstm import ConvLSTM
import torch.optim as optim
import torch.nn as nn
from utils.utils import *


def val():
    # settings
    ckpt = '../ckpt/epoch 4_val_loss_1.0855886936.pth'
    valset = ZebraDataset('../data/val', 'val', window=11, size=128)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                              shuffle=False, num_workers=0)

    model = ConvLSTM(input_dim=3,
                 hidden_dim=[64, 64, 64, 64, 128, 128],
                 kernel_size=[(3, 3),(3, 3),(3, 3),(3, 3),(3, 3),(3, 3)],
                 final_conv_kernel_size=4,
                 num_layers=6,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

    model.load_state_dict(torch.load(ckpt))
    model.eval()

    model = model.cuda()
    criterion = nn.MSELoss()

    # Validation
    val_epoch_loss = AverageMeter()
    with torch.no_grad():
        for index, (inputs, gts) in enumerate(valloader):
            outputs = model(inputs)[0]
            loss = criterion(outputs, gts)
            val_epoch_loss.update(loss.item())
            if index % 100 == 0:  # print every 100 mini-batches
                print('Validating: [iteration %5d / %5d]' %
                      (index + 1, len(valset) // len(inputs)))
    print('Validation avg loss: %.10f' % val_epoch_loss.avg())



if __name__ == '__main__':
    val()