import os
import scipy.io as sio
import numpy as np
from skimage import io
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import data_augmentation as augmentation
import hyper_parameter_loader as hp

class data_loader(data_utils.Dataset):
    def __init__(self):
        self.path = '../data/raw/'+ hp.data_name+'/'+ hp.data_name + '.mat'
        self.total_ins_num = 0
        self.ins_feat_list, self.bag_label_list, self.max_ins_num = self.create_bags()
    def create_bags(self):
        raw_data = sio.loadmat(self.path)
        ins_feat_list = []
        bag_label_list = []
        max_ins_num = 0
        if hp.data_name in hp.bencmark_data or 'NEWS' in hp.data_name:
            for bag_info in raw_data['data']:
                ins_feats = bag_info[0]
                ins_feats = np.delete(ins_feats.T, [-1], 0).T
                bag_label = int(bag_info[1].squeeze())
                if hp.data_name in ['FOX', 'TIGER', 'ELEPHANT']:
                    bag_label = 1 if bag_label < 0 else 0
                if hp.data_name in ['MUSK1', 'MUSK2']:
                    bag_label = 0 if bag_label == 1 else 1
                if 'NEWS' in hp.data_name:
                    ins_feats *= 2
                if len(ins_feats) == 1:
                    ins_feats = np.repeat(ins_feats, hp.ins_pad_num, 0)
                    for i in range(1, hp.ins_pad_num):
                        ins_feats[i] = ins_feats[i] + np.random.randn(ins_feats.shape[-1])/10
                elif len(ins_feats) >= 500:
                    row_sequence = np.arange(0, len(ins_feats), 5)
                    ins_feats = ins_feats[row_sequence, :]
                if len(ins_feats) > max_ins_num:
                    max_ins_num = len(ins_feats)
                self.total_ins_num += len(ins_feats)
                ins_feat_list.append(ins_feats)
                bag_label_list.append(bag_label)
        else:
            ins_feat_dict = raw_data['x'][0][0][0]
            raw_data['x'][0][0][2][raw_data['x'][0][0][2] == 2] = 0
            ins_label_dict = raw_data['x'][0][0][2]
            bag_dict = {}
            for ins_idx, bag_name in enumerate(zip(raw_data['x'][0][0][11][0][0][0], raw_data['x'][0][0][11][0][0][1])):
                bag_name = int(bag_name[1]) - 1
                if bag_name not in bag_dict.keys():
                    bag_dict[bag_name] = []
                bag_dict[bag_name].append(ins_idx)
            for bag_name in bag_dict.keys():
                ins_idxes = bag_dict[bag_name]
                ins_feats = ins_feat_dict[ins_idxes]
                bag_label = ins_label_dict[ins_idxes].squeeze().astype(np.long)[0]
                if len(ins_feats) > max_ins_num:
                    max_ins_num = len(ins_feats)
                self.total_ins_num += len(ins_feats)
                ins_feat_list.append(ins_feats)
                bag_label_list.append(bag_label)
        return ins_feat_list, bag_label_list, max_ins_num
    def __len__(self):
        return len(self.ins_feat_list)
    def __getitem__(self, idx):
        bag_label = self.bag_label_list[idx]
        ins_feats = self.ins_feat_list[idx]
        return bag_label, ins_feats

class COLON_loader(data_utils.Dataset):
    def __init__(self, data_augmentation=False, loc_info=False):
        self.path = '../data/raw/COLON'
        self.data_augmentation = data_augmentation
        self.location_info = loc_info
        self.data_augmentation_img_transform = \
            transforms.Compose([augmentation.RandomHEStain(),
            augmentation.HistoNormalize(),
            augmentation.RandomRotate(),
            augmentation.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.normalize_to_tensor_transform = \
            transforms.Compose([augmentation.HistoNormalize(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dirs = [x[0] for x in os.walk(self.path)]
        dirs.pop(0)
        self.dir_list = dirs
        self.total_ins_num = 0
        self.bag_list, self.ins_label_list, self.bag_label_list, self.max_ins_num = self.create_bags()
    def create_bags(self):
        ins_img_list = []
        bag_label_list = []
        ins_label_list = []
        for dir in self.dir_list:
            img_name = dir.split('/')[-1]
            img_dir = dir + '/' + img_name + '.bmp'
            img = io.imread(img_dir)
            if self.location_info:
                xs = np.arange(0, 500)
                xs = np.asarray([xs for i in range(500)])
                ys = xs.transpose()
                img = np.dstack((img, xs, ys))
            ins_imgs = []
            ins_labels = []
            for label, cell_type in enumerate(['epithelial', 'fibroblast', 'inflammatory', 'others']):
                dir_cell = dir + '/' + img_name + '_' + cell_type + '.mat'
                mat_cell = sio.loadmat(dir_cell)
                for (x, y) in mat_cell['detection']:
                    x = np.round(x)
                    y = np.round(y)
                    if self.data_augmentation:
                        x = x + np.round(np.random.normal(0, 2, 1))
                        y = y + np.round(np.random.normal(0, 2, 1))
                    if x < 13:
                        x_start = 0
                        x_end = 27
                    elif x > 500 - 14:
                        x_start = 500 - 27
                        x_end = 500
                    else:
                        x_start = x - 13
                        x_end = x + 14
                    if y < 13:
                        y_start = 0
                        y_end = 27
                    elif y > 500 - 14:
                        y_start = 500 - 27
                        y_end = 500
                    else:
                        y_start = y - 13
                        y_end = y + 14
                    ins_imgs.append(img[int(y_start):int(y_end), int(x_start):int(x_end)])
                    ins_labels.append(label)
            if len(ins_imgs) == 1:
                ins_imgs = np.repeat(ins_imgs, hp.ins_pad_num, 0)
                for i in range(1, hp.ins_pad_num):
                    ins_imgs[i] = ins_imgs[i] + np.round(np.random.normal(0, 2, 1))
                ins_labels = ins_labels*hp.ins_pad_num
            elif len(ins_imgs) >= 500:
                row_sequence = np.arange(0, len(ins_imgs), 5)
                ins_imgs = np.array(ins_imgs)
                ins_imgs = ins_imgs[row_sequence, :].tolist()
                ins_labels = np.array(ins_labels)
                ins_labels = ins_labels[row_sequence].tolist()
            ins_img_list.append(ins_imgs)
            ins_label_list.append(np.array(ins_labels))
            bag_label = 0 if 0 in ins_labels else 1
            bag_label_list.append(bag_label)
        max_ins_num = 0
        for labels in ins_label_list:
            self.total_ins_num += len(labels)
            if len(labels) > max_ins_num:
                max_ins_num = len(labels)
        return ins_img_list, ins_label_list, bag_label_list, max_ins_num
    def transform_and_data_augmentation(self, bag):
        if self.data_augmentation:
            img_transform = self.data_augmentation_img_transform
        else:
            img_transform = self.normalize_to_tensor_transform
        bag_tensors = []
        for img in bag:
            if self.location_info:
                bag_tensors.append(torch.cat(
                    (img_transform(img[:, :, :3]),
                     torch.from_numpy(img[:, :, 3:].astype(float).transpose((2, 0, 1))).float())))
            else:
                bag_tensors.append(img_transform(img))
        return torch.stack(bag_tensors)
    def __len__(self):
        return len(self.bag_list)
    def __getitem__(self, idx):
        bag_label = self.bag_label_list[idx]
        ins_imgs = self.bag_list[idx]
        ins_labels = self.ins_label_list[idx]
        return bag_label, self.transform_and_data_augmentation(ins_imgs), ins_labels
