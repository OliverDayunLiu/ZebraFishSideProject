import torch
from datasets.zebra_dataset import ZebraDataset
import sys, os
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from utils.utils import *


def train_and_val():
    # settings
    save_epoch_frequency = 1
    save_dir = '../ckpt'
    accumulation_steps = 20
    checkpoint = ''
    #checkpoint = os.path.join(save_dir, 'epoch 21_val_loss_0.0405578714.pth')

    check_dir(save_dir)
    trainset = ZebraDataset('../data/train', 'train', size=256)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    valset = ZebraDataset('../data/val', 'val', size=256)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                              shuffle=False, num_workers=0)

    #model = models.vgg16_bn(pretrained=True)
    #model = models.vgg19_bn(pretrained=True)
    model = models.resnet50(pretrained=True)
    #model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 1)

    #model.classifier[-1] = nn.Linear(in_features=4096, out_features=1024)
    #model.classifier.add_module('relu', nn.ReLU())
    #model.classifier.add_module('last_linear', nn.Linear(in_features=1024, out_features=1))

    if checkpoint != '':
        model.load_state_dict(torch.load(checkpoint))

    model = model.cuda()
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
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
                #print("outputs: " + str(outputs[0][0].item()) + " gts: " + str(gts[0][0].item()))
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