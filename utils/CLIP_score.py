
import torch
from torchvision import transforms
import os
from PIL import Image
from  torchmetrics.multimodal.clip_score import CLIPScore
import re
from torch.utils.data import DataLoader
import json
import pandas as pd
from LT_project.data.load_data import Cifar100Dataset

 #! Description: This file contains the code for filtering the data based on the CLIP score and FID score


def transform_clip(img):
    tf = transforms.Compose([transforms.ToTensor()])
    return tf(img)

def check_CLIP_score_per_image(image,text,device):
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    img_tensor = transform_clip(image).to(device)
    score =  metric(img_tensor,text).detach().cpu().numpy()
    return score

def CLIP_score(image_path,text_path,device): #돌아가는거 check완료 
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    text_dict = json.load(open(text_path,'r'))
    score_dict ={}
    
    for folder in os.listdir(image_path): #! main path
        files = os.listdir(os.path.join(image_path,folder)) #! subfolder
        score_list = []
        for file in files:
            image_name = os.path.join(image_path,folder,file)
            k_idx,t_idx ,img_num= image_name.split('_')[-3], int(image_name.split('_')[-2][4:]),int(image_name.split('_')[-1].split('.')[0])
            text = text_dict[k_idx][t_idx]
            img = Image.open(image_name)
            print('image get')
            img_tensor = transform_clip(img).to(device) #여기가 필수 였네.... 
            score = metric(img_tensor,text).detach().cpu().numpy()
            score_list.append(score)
            print(f'text_idx : {t_idx} img_num : {img_num} CLIP score :{score}')
        score_dict[k_idx] = score_list 
        print('folder finish')
        df = pd.DataFrame.from_dict(score_dict,orient='index')
        df.to_csv('./csvfile/clip_score1.csv')
    return score_dict


#.................. CLIP csv file processing..................#
def extract_clip_score(text):
    clip_score_match = re.search(r'CLIP score :([0-9.]+)',text)
    if clip_score_match:
        return round(float(clip_score_match.group(1)),3)
    return None

def make_csv_preprocessed_file(csv_file_path):
    df = pd.read_csv(csv_file_path,index_col = 0)
    for row in df.index:    
        for col in df.columns:
            df.at[row, col] = extract_clip_score(df.at[row, col])
    df.to_csv('./csvfile/clip_score_processed.csv')



# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# image_path= '../datasets/changed_syn_cifar100'
# text_path = './LT_project/cifar100_description.json'
# _ = torch.manual_seed(42)







