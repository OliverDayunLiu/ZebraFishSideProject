import os, sys


class AverageMeter:
    def __init__(self):
       self.sum = 0
       self.count = 0

    def avg(self):
        return self.sum/self.count

    def update(self, sum):
        self.sum += sum
        self.count += 1

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)