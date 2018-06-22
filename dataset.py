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

# Refer to https://arxiv.org/pdf/1603.00831.pdf
Label = [
    'Background',
    'Pedestrian',              # 1
    'Person on vehicle',
    'Car',                     # 3
    'Bicycle',
    'Motorbike',
    'Non motorized vehicle',
    'Static person',           # 7
    'Distractor',
    'Occluder',
    'Occluder on the ground',
    'Occluder full',           # 11
    'Reflection'
    ]

# Ground Truth file format, 1-based
'''
1 Frame number Indicate at which frame the object is present
2 Identity number Each pedestrian trajectory is identified by a unique ID (−1 for detections)
3 Bounding box left Coordinate of the top-left corner of the pedestrian bounding box
4 Bounding box top Coordinate of the top-left corner of the pedestrian bounding box
5 Bounding box width Width in pixels of the pedestrian bounding box
6 Bounding box height Height in pixels of the pedestrian bounding box
7 Confidence score DET: Indicates how confident the detector is that this instance is a pedestrian.
  GT: It acts as a flag whether the entry is to be considered (1) or ignored (0).
8 Class GT: Indicates the type of object annotated
9 Visibility GT: Visibility ratio, a number between 0 and 1 that says how much of that object is visible. 
  Can be due to occlusion and due to image border cropping.
'''

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
        # get ground truth
        labels = []
        fp = os.path.join(self.root, v, 'gt', 'gt.txt')
        if os.path.exists(fp):
            df = pd.read_csv(fp, header=None)
            df = df.loc[ df[0] == idx+1 ] # column 0 is Frame number
            for r in df.itertuples():
                l = {
                    'id': r._2,
                    'rect': [r._3, r._4, r._5, r._6],
                    'show': r._7,
                    'class': r._8,
                    'ratio': r._9,
                    }
                labels.append(l)
        # pack as sample
        sample = {'image': img, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    train = MoTDataset('data/MOT17Det/train')
    idx = random.randint(0, len(train)-1)
    sample = train[idx]
    # display original image
    sample['image'].show()
    #print(sample['label'])