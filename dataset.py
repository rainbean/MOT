import os
import random
import json
import glob
import numpy as np
import pandas as pd
import configparser
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tx
from PIL import Image, ImageOps, ImageDraw
from helper import config

# Ignore skimage convertion warnings
import warnings
warnings.filterwarnings("ignore")

class MoTDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.seq = self.parse(root)

    def parse(self, root):
        ''' iterate seqinfo.ini of each subfoler, then validate existing of real files '''
        seq = {}
        seqinfo = configparser.ConfigParser()        
        for v in os.listdir(root):
            seqinfo.read(os.path.join(root, v, 'seqinfo.ini'))
            vlen = seqinfo['Sequence'].getint('seqlength', 0)
            if not vlen:
                continue
            vext = seqinfo['Sequence']['imExt']
            fn = len(glob.glob(os.path.join(root, v, 'img1', '*' + vext)))
            if fn != vlen:
                print('WARNING: {} - amound of frame files {} mismatch with seqinfo {}'.format(v, fn, vlen))
                continue
            seq[v] = {'len': vlen, 'ext': vext}
        return seq

    def __len__(self):
        return sum([ self.seq[v]['len'] for v in self.seq ])

    def __getitem__(self, idx):
        # get subidx of video seq
        for v in self.seq:
            vlen = self.seq[v]['len']
            if idx < vlen:
                break
            idx -= vlen
        # decode frame
        fp = os.path.join(self.root, v, 'img1', "{:06d}{}".format(idx+1, self.seq[v]['ext']))
        try:
            img = Image.open(fp)
        except:
            print('DEBUG: {} - no such file'.format(fp))
            raise IndexError()
        if img.mode != 'RGB':
            img = image.convert('RGB')
        # get labels
        sample = {'image': img }
        # .... 
        
        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    train = MoTDataset('data/MOT17Det/train')
    idx = random.randint(0, len(train)-1)
    sample = train[idx]
    # display original image
    sample['image'].show()