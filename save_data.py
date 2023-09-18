from LT_project.data.imbalance_cifar import IMBALANCECIFAR100

import os
import torch
import torchvision.transforms as transforms
from Data_synthesizing import syn_to_lt_folder,init_dataset

#! only for making imbalance dataset if the dataset doesn't exist
# Instantiate the custom dataset
train_dataset = IMBALANCECIFAR100(root='./datasets', imb_type='exp', imb_factor=0.01, rand_number=0, train=True, transform=None, download=True)
syn_image_path = './datasets/changed_syn_cifar100'
text_path = './LT_project/json/cifar100_description_1.json' 
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


#! making the datset
dist = [200,300,400]
# Specify the directory to save the dataset
for d in dist:
    save_dir = f'./datasets/cifar100-lt/train_0.01_{d}'  # Change this to your desired directory
    os.makedirs(save_dir, exist_ok=True)

# Create subfolders for each class and save images
    for class_idx, class_name in enumerate(train_dataset.classes):
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    # Find indices of images belonging to this class
        indices = [i for i, label in enumerate(train_dataset.targets) if label == class_idx]

    # Copy images to the class subfolder
        for idx in indices:
            img, _ = train_dataset[idx]  # Get image and target
            img_filename = f"{idx:05d}.png"  # Adjust the filename format if needed
            img_path = os.path.join(class_dir, img_filename)
            img.save(img_path)

    print("Dataset saved into subfolders successfully.")

# replace the subfolder name if name has '_' into ' 'blank
    for root, dirs, files in os.walk(save_dir):
        for dir in dirs:
            if '_' in dir:
                new_dir = dir.replace('_', ' ')
                src = os.path.join(root, dir)
                dst = os.path.join(root, new_dir)
                os.rename(src, dst)
            
    print('Eliminate _ in the subfolder name')
    
    if 'noCLIP' in d:
        check = False
        syn_to_lt_folder(syn_image_path,save_dir,text_path,int(d.split('_')[0]),device,CLIP = check) #image 채워넣기
        print(f'Synthesized images are added in {d}set')
    else:
        check = True
        syn_to_lt_folder(syn_image_path,save_dir,text_path,int(d),device,CLIP = check) #image 채워넣기
        print(f'Synthesized images are added in {d}set')

# init_dataset(lt_img_path) #erase all the added images
    