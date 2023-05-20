import torch
from datasets.zebra_dataset_lstm import ZebraDatasetLSTM
import sys, os
from models.convlstm import ConvLSTM
import torch.optim as optim
import torch.nn as nn
from utils.utils import *


def train_and_val():
    # settings
    save_epoch_frequency = 1
    save_dir = '../ckpt'
    accumulation_steps = 20

    check_dir(save_dir)
    trainset = ZebraDatasetLSTM('../data/train', 'train', window=11, size=128)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    valset = ZebraDatasetLSTM('../data/val', 'val', window=11, size=128)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1,
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
    model = model.cuda()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    epoch = 0
    while True:
        # Train
        model.train()
        train_epoch_loss = AverageMeter()
        for index, (inputs, gts) in enumerate(trainloader):
            outputs = model(inputs)
            loss = criterion(outputs, gts)
            loss = loss / accumulation_steps
            loss.backward()

            if (index+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics
            train_epoch_loss.update(loss.item())
            if index % 100 == 0:  # print every 100 mini-batches
                print('Training: [epoch %d, iteration %5d / %5d] avg loss: %.10f' %
                      (epoch + 1, index + 1, len(trainset)//len(inputs), train_epoch_loss.avg()))

        # Validation
        model.eval()
        val_epoch_loss = AverageMeter()
        with torch.no_grad():
            for index, (inputs, gts) in enumerate(valloader):
                outputs = model(inputs)
                loss = criterion(outputs, gts)
                #print("outputs: " + outputs + " gts: " + gts)
                # print statistics
                val_epoch_loss.update(loss.item())
                if index % 100 == 0:  # print every 100 mini-batches
                    print('Validating: [iteration %5d / %5d]' %
                          (index + 1, len(valset) // len(inputs)))
        print('epoch %d Validation avg loss: %.10f' % (epoch + 1, val_epoch_loss.avg()))


        epoch += 1
        if epoch % save_epoch_frequency == 0:
            save_path = os.path.join(save_dir, 'epoch ' + str(epoch) + '_val_loss_' + '%.10f' % val_epoch_loss.avg() + '.pth')
            torch.save(model.state_dict(), save_path)
            print("model saved")



if __name__ == '__main__':
    train_and_val()