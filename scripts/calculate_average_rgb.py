import sys, os
from utils.utils import *
import numpy as np
import cv2


def calculate_average_rgb():
    imgs_folder = '../data/train/imgs'
    for drug_name_folder in os.listdir(imgs_folder):
        drug_name_folder_fullname = os.path.join(imgs_folder, drug_name_folder)
        if not os.path.isdir(drug_name_folder_fullname):
            continue
        averageB, averageG, averageR = AverageMeter(), AverageMeter(), AverageMeter()
        for filename in os.listdir(drug_name_folder_fullname):
            if '.jpg' not in filename and '.png' not in filename:
                continue
            full_img_path = os.path.join(drug_name_folder_fullname, filename)
            img = cv2.imread(full_img_path)
            if img is None:
                print("reading img failed: ", full_img_path)
            averageB.update(np.mean(img[:, :, 0]))
            averageG.update(np.mean(img[:, :, 1]))
            averageR.update(np.mean(img[:, :, 2]))
        print("%s has average RGB: %.2f %.2f %.2f" % (drug_name_folder, averageR.avg(), averageG.avg(), averageB.avg()))


if __name__ == '__main__':
    calculate_average_rgb()