import copy
import os

import pandas as pd
import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image,ImageOps
from tifffile import imread
import random

class OriDataset(Dataset):

    def __init__(self,configs):  # crop_size,

        cell_type=configs['cell_type']
        purpose = configs['purpose']
        #self.add_mask = configs['add_mask']
        cutoff = 0.5
        data_root = configs['dir_paths']
        image_size = configs['image_size']

        a549 = ['BR00116991__2020-11-05T19_51_35-Measurement1/',
                'BR00116992__2020-11-05T21_31_31-Measurement1/',
                'BR00116993__2020-11-05T23_11_39-Measurement1/',
                'BR00116994__2020-11-06T00_59_44-Measurement1/', ]
        u2os = ['BR00116995__2020-11-06T02_41_05-Measurement1/',
                'BR00117024__2020-11-06T04_20_37-Measurement1',
                'BR00117025__2020-11-06T06_00_19-Measurement1',
                'BR00117026__2020-11-06T07_39_45-Measurement1',
                ]

        if cell_type == 'u2os':
            if purpose == 'train' or purpose == 'val':
                subfolders = u2os[:3]
            if purpose == 'test':
                subfolders = a549
        elif cell_type == 'a549':
            if purpose == 'train' or purpose == 'val':
                subfolders = a549[:3]
            if purpose == 'test':
                subfolders = u2os
        elif cell_type == 'both':
            if purpose == 'train' or purpose == 'val':
                subfolders = a549[:2] + u2os[:2]
            if purpose == 'test':
                subfolders = a549[3:] + u2os[3:]


        # subfolders = ['BR00118041__2020-11-05T03_24_00-Measurement1',
        #               'BR00118045__2020-11-03T05_49_34-Measurement1']

        self.imlist = []

        self.pixel_cutoff = cutoff
        self.purpose = purpose
        self.filepath = data_root
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
        ])
        for folder in subfolders:
            fpath = os.path.join(data_root, folder, 'Images')
            flist = os.listdir(fpath)
            flist = [a.split('-')[0] for a in flist]
            flist = list(np.unique(flist))
            flist.remove('Index.idx.xml')
            self.imlist += [os.path.join(fpath, file) for file in flist]
        self.imlist = self.imlist
        if purpose == 'val':
            self.imlist = self.imlist[:10]

    def normalize(self, img):

        lower_threshold = np.percentile(img, self.pixel_cutoff)
        upper_threshold = np.percentile(img, 100 - self.pixel_cutoff)

        img_clipped = np.clip(img, lower_threshold, upper_threshold)

        img_clipped = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped)) * 255

        return np.array(img_clipped, dtype=np.uint8)

    def __getitem__(self, idx):
        file = self.imlist[idx]
        bf = []
        # for i in [6, 7, 8]:

        for i in [6,6, 6, 6, 6]:
            img = imread(file + '-ch%isk1fk1fl1.tiff' % (i))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            bf.append(img_preprocessed)

        label_tfm = torch.cat(bf)

        # others
        output = []
        for i in range(1, 6):
        # for i in [3,4,5]:
            img = imread(os.path.join(file + '-ch%isk1fk1fl1.tiff' % (i)))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            output.append(img_preprocessed)

        img_tfm = torch.cat(output)

        if self.purpose == 'test':
            return_d = {'gt': img_tfm, 'lq': label_tfm, 'filename': file
                        }
        else:
            return_d = {'gt': img_tfm, 'lq': label_tfm,
                        }
        return return_d
    def __len__(self):
        return len(self.imlist)

class OriDataset_3channel(Dataset):

    def __init__(self,configs):  # crop_size,

        cell_type=configs['cell_type']
        purpose = configs['purpose']
        #self.add_mask = configs['add_mask']
        cutoff = 0.5
        data_root = configs['dir_paths']
        image_size = configs['image_size']

        a549 = ['BR00116991__2020-11-05T19_51_35-Measurement1/',
                'BR00116992__2020-11-05T21_31_31-Measurement1/',
                'BR00116993__2020-11-05T23_11_39-Measurement1/',
                'BR00116994__2020-11-06T00_59_44-Measurement1/', ]
        u2os = ['BR00116995__2020-11-06T02_41_05-Measurement1/',
                'BR00117024__2020-11-06T04_20_37-Measurement1',
                'BR00117025__2020-11-06T06_00_19-Measurement1',
                'BR00117026__2020-11-06T07_39_45-Measurement1',
                ]

        if cell_type == 'u2os':
            if purpose == 'train' or purpose == 'val':
                subfolders = u2os[:3]
            if purpose == 'test':
                subfolders = a549
        elif cell_type == 'a549':
            if purpose == 'train' or purpose == 'val':
                subfolders = a549[:3]
            if purpose == 'test':
                subfolders = u2os
        elif cell_type == 'both':
            if purpose == 'train' or purpose == 'val':
                subfolders = a549[:2] + u2os[:2]
            if purpose == 'test':
                subfolders = a549[3:] + u2os[3:]

        self.imlist = []

        self.pixel_cutoff = cutoff
        self.purpose = purpose
        self.filepath = data_root
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
        ])
        for folder in subfolders:
            fpath = os.path.join(data_root, folder, 'Images')
            flist = os.listdir(fpath)
            flist = [a.split('-')[0] for a in flist]
            flist = list(np.unique(flist))
            flist.remove('Index.idx.xml')
            self.imlist += [os.path.join(fpath, file) for file in flist]
        self.imlist = self.imlist
        if purpose == 'val':
            self.imlist = self.imlist[:10]

    def normalize(self, img):

        lower_threshold = np.percentile(img, self.pixel_cutoff)
        upper_threshold = np.percentile(img, 100 - self.pixel_cutoff)

        img_clipped = np.clip(img, lower_threshold, upper_threshold)

        img_clipped = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped)) * 255

        return np.array(img_clipped, dtype=np.uint8)

    def __getitem__(self, idx):
        file = self.imlist[idx]
        bf = []
        # for i in [6, 7, 8]:

        for i in [6, 7, 8]:
            img = imread(file + '-ch%isk1fk1fl1.tiff' % (i))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            bf.append(img_preprocessed)

        label_tfm = torch.cat(bf)

        # others
        output = []
        # for i in range(1, 6):
        for i in [1,2,5]:
            img = imread(os.path.join(file + '-ch%isk1fk1fl1.tiff' % (i)))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            output.append(img_preprocessed)

        img_tfm = torch.cat(output)

        if self.purpose == 'test':
            return_d = {'gt': img_tfm, 'lq': label_tfm, 'filename': file
                        }
        else:
            return_d = {'gt': img_tfm, 'lq': label_tfm,
                        }
        return return_d
    def __len__(self):
        return len(self.imlist)


def process_edge_map_to_instance_mask(edge_map):
    """
    Converts a binary edge map to an instance segmentation map.

    Parameters:
    edge_map (numpy.ndarray): Binary edge map.

    Returns:
    numpy.ndarray: Instance segmentation map with unique labels for each object.
    """
    # Ensure the edge map is binary
    edge_map = (edge_map > 0).astype(np.uint8) * 255

    # Apply morphological operations (optional, based on the quality of the edge map)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask
    instance_mask = np.zeros_like(edge_map, dtype=np.int32)

    # Draw each contour with a unique label
    for i, contour in enumerate(contours):
        cv2.drawContours(instance_mask, [contour], -1, (i + 1), thickness=cv2.FILLED)


    return instance_mask



class OriDataset_wmask(Dataset):

    def __init__(self,configs):  # crop_size,

        cell_type=configs['cell_type']
        purpose = configs['purpose']
        self.purpose = purpose
        #self.add_mask = configs['add_mask']
        cutoff = 0.5
        data_root = configs['dir_paths']
        image_size = configs['image_size']

        a549 = ['BR00116991__2020-11-05T19_51_35-Measurement1/',
                'BR00116992__2020-11-05T21_31_31-Measurement1/',
                'BR00116993__2020-11-05T23_11_39-Measurement1/',
                'BR00116994__2020-11-06T00_59_44-Measurement1/', ]
        u2os = [
            'BR00116995__2020-11-06T02_41_05-Measurement1/',
                'BR00117024__2020-11-06T04_20_37-Measurement1',
                'BR00117025__2020-11-06T06_00_19-Measurement1',
                'BR00117026__2020-11-06T07_39_45-Measurement1',
                ]

        if cell_type == 'u2os':
            if purpose == 'train' or purpose == 'val':
                subfolders = u2os[:3]
            if purpose == 'test':
                subfolders = a549
        elif cell_type == 'a549':
            if purpose == 'train' or purpose == 'val':
                subfolders = a549[:3]
            if purpose == 'test':
                subfolders = u2os
        elif cell_type == 'both':
            if purpose == 'train' or purpose == 'val':
                subfolders = a549[:2] + u2os[:2]
            if purpose == 'test':
                subfolders = a549[3:] + u2os[3:]
                # subfolders =  a549[3:]

        self.mask_path = '/media/NAS06/cell_painting/masks/'
        self.imlist = []

        self.pixel_cutoff = cutoff

        self.filepath = data_root
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            # transforms.Normalize([0.5], [0.5]),
        ])
        for folder in subfolders:
            fpath = os.path.join(data_root, folder, 'Images')
            fpath_mask = os.path.join(self.mask_path, folder)
            flist_mask = os.listdir(fpath_mask)
            flist_mask = [a.split('-')[0] for a in flist_mask]
            flist = list(np.unique(flist_mask))


            self.imlist += [os.path.join(fpath, file) for file in flist]

        if purpose == 'val':
            self.imlist = self.imlist[:10]

    def normalize(self, img):

        lower_threshold = np.percentile(img, self.pixel_cutoff)
        upper_threshold = np.percentile(img, 100 - self.pixel_cutoff)

        img_clipped = np.clip(img, lower_threshold, upper_threshold)

        img_clipped = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped)) * 255

        return np.array(img_clipped, dtype=np.uint8)

    def __getitem__(self, idx):
        file = self.imlist[idx]
        bf = []
        # for i in [6, 7, 8]:

        for i in [6,6,6,6, 6, 7, 8]:
            img = imread(file + '-ch%isk1fk1fl1.tiff' % (i))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            bf.append(img_preprocessed)

        label_tfm = torch.cat(bf)

        # others
        output = []
        for i in range(1, 6):
        # for i in [3,4,5]:
            img = imread(os.path.join(file + '-ch%isk1fk1fl1.tiff' % (i)))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            output.append(img_preprocessed)
        file_name = os.path.split(file)[-1]
        plateid = file.split('/')[-3]
        for i in [9, 10]:
            img = Image.open(os.path.join(self.mask_path, plateid, file_name + '-ch%isk1fk1fl1.png' % (i)))
            img = process_edge_map_to_instance_mask(np.array(img))
            img = Image.fromarray(img)
            img_preprocessed = self.transform(img)
            output.append(img_preprocessed)

        img_tfm = torch.cat(output)

        if self.purpose == 'test':
            return_d = {'gt': img_tfm, 'lq': label_tfm, 'filename': file
                        }
        else:
            return_d = {'gt': img_tfm, 'lq': label_tfm,
                        }
        return return_d
    def __len__(self):
        return len(self.imlist)



if __name__ == '__main__':
    dataset = OriDataset('/media/xiaodan/cellpainting/all_images')
    a = dataset[0]
    a = 1
