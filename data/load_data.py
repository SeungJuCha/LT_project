import os
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from PIL import Image
from data.cifar100_classes import Cifar100Class
import torch


def cifar100_labeling():
    class_dict = Cifar100Class
    return class_dict

def transform(mode=None): #from LDAM-DRW transfromation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if mode =='train':
        return transform_train
    elif mode =='val':
        return transform_val
    elif mode is None:
        return transform
    else:
        raise ValueError('mode should be train or val or None')
    
class Cifar100Dataset(Dataset): #Can load data with subfolder name as label or real-class label
    def __init__(self, root_dir, transform=None, labeling=cifar100_labeling()): 
        self.root_dir = root_dir
        self.transform = transform
        self.label = labeling
        self.class_folders = os.listdir(root_dir) #dataset/cifar-100-python/train_image
        self.image_list = [] # same role in ImagepathDataset in fid-score

        for class_folder in self.class_folders:
            class_path = os.path.join(root_dir, class_folder)
            images = os.listdir(class_path)
            self.image_list.extend([(class_folder, img) for img in images])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        class_folder, img_name = self.image_list[idx] #image list 자체에 tuple 형태 저장(folder_name, img)
        img_path = os.path.join(self.root_dir, class_folder, img_name)
        image = Image.open(img_path)
        # label = self.label[class_folder]  
        if self.label.get(class_folder) is not None:
        # Case 1: class_folder is the name of the class
            label = self.label[class_folder]
        else:
        # Case 2: class_folder is the label of the class
            label = int(class_folder)
        if self.transform:
            image = self.transform(image)

        return image, label


