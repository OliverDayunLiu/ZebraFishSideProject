import numpy as np
from PIL import Image
import sys, os
import pandas
import torch
import random
import torchvision

class ZebraDataset:
    def __init__(self, root_dir, mode, size, imgs_path=''):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'imgs')
        self.mode = mode
        self.img_size = size

        self.img_paths = []
        self.heartrates = []
        if self.mode == 'train' or self.mode == 'val':
            for filename in os.listdir(self.root_dir):
                if '.xlsx' not in filename and '.excel' not in filename:
                    continue
                video_name = filename.split('.')[0]
                full_xml = os.path.join(self.root_dir, filename)
                xml = pandas.read_excel(full_xml)
                atrium_values = xml['Atrium'].tolist()
                atrium_values = np.array(atrium_values)
                largest_atrium_value = np.max(atrium_values)
                smallest_atrium_value = np.min(atrium_values)
                frame_nums = np.arange(1, len(atrium_values)+1)
                for i in range(0, len(frame_nums)):
                    frame_num = int(frame_nums[i])
                    img_prefix = '%05d' % frame_num
                    img_path = os.path.join(self.img_dir + '/' + video_name, img_prefix + '.png')
                    self.img_paths.append(img_path)
                    value = float(atrium_values[i])
                    self.heartrates.append((2*(value-smallest_atrium_value))/(largest_atrium_value-smallest_atrium_value)-1)
        else:
            if imgs_path != '':
                for filename in os.listdir(imgs_path):
                    if '.jpg' not in filename and '.png' not in filename:
                        continue
                    self.img_paths.append(os.path.join(imgs_path, filename))

        # Data augmentation
        if self.mode == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(self.img_size),
                torchvision.transforms.RandomHorizontalFlip(),
                #torchvision.transforms.RandomRotation(20),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                #torchvision.transforms.Normalize([0.406, 0.406, 0.406], [0.225, 0.225, 0.225])
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.img_size, self.img_size)),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                #torchvision.transforms.Normalize([0.406, 0.406, 0.406], [0.225, 0.225, 0.225])
            ])

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = Image.open(img_name)
        img = img.convert("RGB")
        img = self.transform(img).cuda()
        #img[0, :, :] = img[2, :, :]
        #img[1, :, :] = img[2, :, :]
        if self.mode != 'test':
            atrium_value = self.heartrates[index]
            gt = torch.Tensor([atrium_value]).float().cuda()
            return img, gt
        else:
            return img

    def __len__(self):
        return len(self.img_paths)
