import numpy as np
import sys, os
import pandas
import torch
import random
from PIL import Image
import torchvision

class ZebraDatasetLSTM:
    def __init__(self, root_dir, mode, window, size):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'imgs')
        self.mode = mode
        self.window = window # window of frames for LSTM time dimension
        self.img_size = size

        self.img_paths = []
        self.heartrates = []
        for filename in os.listdir(self.root_dir):
            if '.xlsx' not in filename and '.excel' not in filename:
                continue
            which_video = filename.split('.')[0]
            full_xml = os.path.join(self.root_dir, filename)
            xml = pandas.read_excel(full_xml)
            atrium_values = xml['Atrium'].tolist()
            frame_nums = np.arange(1, len(atrium_values)+1)

            for i in range(0+self.window//2, len(frame_nums)-self.window//2):
                frame_num = int(frame_nums[i])
                img_prefix = '%05d' % frame_num
                img_path = os.path.join(self.img_dir + '/' + which_video, img_prefix + '.png')
                self.img_paths.append(img_path)
                self.heartrates.append(float(atrium_values[i]))

        # Data augmentation
        if self.mode == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.img_size, self.img_size)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(20),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.img_size, self.img_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __getitem__(self, index):
        middle_img_name = self.img_paths[index]
        middle_img_folder = os.path.split(middle_img_name)[0] # '../data/train\\imgs/1'
        middle_img_filename = os.path.split(middle_img_name)[1] # '00226.jpg'
        middle_frame_num = int(middle_img_filename.split('.')[0])
        start_frame_num = middle_frame_num - self.window//2
        frames = []
        for i in range(0, self.window):
            curr_num = start_frame_num + i
            img_name = os.path.join(middle_img_folder, '%05d' % curr_num + '.png')
            img = Image.open(img_name)
            if img is None:
                print("read img failed: ", img_name)
            img = img.convert("RGB")
            img = self.transform(img).cuda()
            frames.append(img)
        atrium_value = self.heartrates[index]

        input = torch.stack(frames, dim=0)
        gt = torch.Tensor([atrium_value]).float().cuda()
        return input, gt

    def __len__(self):
        return len(self.img_paths)
