import os
import torch
import json
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler # check the scheduler you want to use
import numpy as np
from PIL import Image
from data.cifar100_classes import Cifar100Class, Coarse_label
from torchvision.transforms import ToTensor
from utils.CLIP_score import check_CLIP_score_per_image,transform_clip
from collections import Counter
from  torchmetrics.multimodal.clip_score import CLIPScore
from data.load_data import Cifar100Dataset,transform
from collections import defaultdict
import pandas as pd
import torch.nn as nn
# from multiprocessing import Process

def calculate_class_counts(dataset):
    class_counts = defaultdict(int)
    
    for idx in range(len(dataset)):
        _, label = dataset[idx] # image, label
        class_counts[label] += 1
    
    sort_class_counts = dict(sorted(class_counts.items()))
    return sort_class_counts



def open_file(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file


#? This part is for generating image from each prompt

    #? model loading
def load_model(model_id = None, pipeline = None, scheduler = None,device= None):
    if device is None:
        device = torch.device('cpu')
    else:
        device = device
    
    if model_id is None :
        model_id = 'runwayml/stable-diffusion-v1-5'
    else:
        model_id = model_id
    if pipeline is None:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype = torch.float16) #memory efficient use float16
        pipe.to(device)
    else:
        pipe = pipeline.from_pretrained(model_id,torch_dtype = torch.float16) #memory efficient use float16
        pipe.to(device)
    #? generator loading
    if scheduler is None:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    return pipe

#? This part is for generate the number of image from each prompt with steps
def get_inputs(promptlist, batch_size=1, num_inference_steps=None, device='cpu'):
    output_prompt = []
    generators = [torch.Generator(device=device).manual_seed(i) for i in range(batch_size)]
    
    for prompt in promptlist:
        prompts = batch_size * [prompt]
        output_prompt.append(prompts)
    
    if num_inference_steps is not None:
        num_inference_steps = num_inference_steps
    else:
        num_inference_steps = 50
    
    return {'prompts': output_prompt, 'num_inference_steps': num_inference_steps, 'generators': generators}

def generate_image(output_path,prompt_file_path, model_id,pipeline,scheduler,batch_size,num_inference_steps,filter = False, device = 'cpu'):
    
    """input example #! original
    output_path : where to save the generated image large folder
    prompt_file_path : json file path with {class name : prompt list or prompt}
    model_id : model id for loading the model in hugging face hub
    pipeline : pipeline for loading the model
    scheduler : scheduler for loading the model
    batch_size : number of image to generate per prompt (generator)
    num_inference_steps : steps
    filter : boolean value , wheter to filter the image with CLIP score or not
    device : device to use (cpu or cuda)"""
    
    file = open_file(prompt_file_path)
    pipe = load_model(model_id=model_id,pipeline=pipeline,scheduler=scheduler,device=device)
    for key in list(file.keys())[:1]: # 생성하는데 굉장히 오래 걸리기에 끊어서 추천 
        
        prompt_list = file[key]
        config_dict = get_inputs(promptlist=prompt_list,batch_size=batch_size,num_inference_steps=num_inference_steps,device = device)
        p = config_dict['prompts']  
        g = config_dict['generators']
        n = config_dict['num_inference_steps']
        gen_path = output_path + f'/{key}'  # Replace with the actual path
        os.makedirs(gen_path, exist_ok=True)  # Create the output directory if it doesn't exist

        for idx, text in enumerate(p):
            # for text in text_list:
            images = pipe(prompt=text, num_inference_steps=n, generator=g).images
            for i, image_pil in enumerate(images):
#................filtering part when generating images...............#
                if filter:
                    print('Filtering is running...')
                    score = check_CLIP_score_per_image(image_pil,text[i],device)
                    if score >= 25: #! threshold
                        image_filename = os.path.join(gen_path, f'image_{key}_text{idx}_{i}.png')
                        image_pil.save(image_filename)
                        print(f'Saved: {image_filename}')
                        torch.cuda.empty_cache()
                else:
                    print('Save all the generated image')
                    image_filename = os.path.join(gen_path, f'image_{key}_text{idx}_{i}.png')
                    image_pil.save(image_filename)
                    print(f'Saved: {image_filename}')
                    torch.cuda.empty_cache()
        print(key,'is done')
    print('Image Gneration is done!')
    

#! this part is checking CLIP socre of each image & prompt -> threshold -> transform into 32,32 and save it in train_0.01 folder(label subfolder)
def syn_to_lt_folder(syn_image_path,lt_img_path,text_path,distribution,device,CLIP = True):
    csv_path = './csvfile/clip_score_processed.csv'
    data = pd.read_csv(csv_path)
    class_mean_dict = {}
    for _, row in data.iterrows():
        class_label = row['Unnamed: 0']
        class_mean = row['Mean']
        class_mean_dict[class_label] = class_mean
        
    
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    text_dict = json.load(open(text_path,'r'))
    origin_dic = calculate_class_counts(Cifar100Dataset(root_dir=lt_img_path, transform=transform())) #! label: count
    for folder in os.listdir(syn_image_path): #! main path   str (class name)
        lack_imagenum = int(distribution- origin_dic[Cifar100Class[folder]])
        threshold = int(class_mean_dict[folder]) #! the mean value of all the image from CLIP Score
        if CLIP == True:
            if lack_imagenum >0:
                print('lack image number : ',lack_imagenum)
                files = os.listdir(os.path.join(syn_image_path,folder)) #! subfolder
                i = 0 #! chaging the image name
                for file in files:
                    image_name = os.path.join(syn_image_path,folder,file)
                    k_idx,t_idx ,_= image_name.split('_')[-3], int(image_name.split('_')[-2][4:]),int(image_name.split('_')[-1].split('.')[0])
            #k_idx : class name
                    text = text_dict[k_idx][t_idx]
                    img = Image.open(image_name)
                    img_tensor = transform_clip(img).to(device) 
                    score = metric(img_tensor,text).detach().cpu().numpy()
                    print(f'CLIP score : {score}')
                    if score >=threshold:
                        img =img.resize((32,32)) #! resizing into cifar100 size
                        img.save(os.path.join(lt_img_path,k_idx,f'image_plus{i}.png'))
                        print(f'image saved in {k_idx}')
                        i+=1
                        lack_imagenum -=1
                        if lack_imagenum == 0:
                            break
            else:
                print('No more image needed go to next folder')
        else:
            if lack_imagenum >0:
                print('lack image number : ',lack_imagenum)
                files = os.listdir(os.path.join(syn_image_path,folder)) #! subfolder 
                i = 0
                for file in files:
                    image_name = os.path.join(syn_image_path,folder,file)
                    k_idx,t_idx ,_= image_name.split('_')[-3], int(image_name.split('_')[-2][4:]),int(image_name.split('_')[-1].split('.')[0])
                    img = Image.open(image_name)
                    img =img.resize((32,32)) #! resizing into cifar100 size
                    img.save(os.path.join(lt_img_path,k_idx,f'image_plus{i}.png'))
                    print(f'image saved in {k_idx}')
                    i+=1
                    lack_imagenum -=1
                    if lack_imagenum == 0:
                        break
            else:
                print('No more image needed go to next folder')


#...................................init dataset.................................#
def init_dataset(lt_img_path, check_name = 'image_plus'):
    for folder in os.listdir(lt_img_path):
        folder_path = os.path.join(lt_img_path, folder)  # Full path to the current folder
        if os.path.isdir(folder_path):  # Check if it's a directory
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)  # Full path to the current file
                if check_name in file and os.path.isfile(file_path):
                    os.remove(file_path)  # Remove the file
                    print(f"Removed: {file_path}")








# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# # #! excute code
# out_dir = '../datasets/syn_cifar100'
# os.makedirs(out_dir, exist_ok=True)
# cifar100_prompts_path = './json/cifar100_description_1.json' #! path for your own prompt file (json)
# # # #! Main excute code
# generate_image(out_dir,cifar100_prompts_path,None,None,None,1,30,False,device) 
