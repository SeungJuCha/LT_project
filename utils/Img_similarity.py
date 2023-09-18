import os
#want to extract feature of image using CLIP image encoder
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel,AutoProcessor
import seaborn as sns
import matplotlib.pyplot as plt
from cifar_train import get_class_num_list,transform
from data.load_data import Cifar100Dataset
import numpy as np

 #ToDo: feature ->1,512  작업중 ..............

torch.cuda.empty_cache()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# Load pretrained model and processor
# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# Load the CLIP tokenizer
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
# Move the model to the device (GPU or CPU)
model.to(device)
# Load the image and preprocessing
path = './datasets/cifar100-lt/train_0.01'
# foldername = 'tank'
tf = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

dataset = Cifar100Dataset(root_dir=path, transform=transform())
class_num_dic = get_class_num_list(dataset)
batch_size = 5
save_path = './LT_projcect/feature_np'
# total = int(max(class_num_dic.values())//batch)


for foldername, image_num in class_num_dic.items():
    subfolder = os.path.join(path, foldername)
    image_paths = [os.path.join(subfolder, file) for file in os.listdir(subfolder)]
    
    batch_img_f_list = []  # List to store image features for a batch
    # batch_mean_img_f_list = []  # List to store batch mean image features
    
    for idx, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path)
        image = tf(image).unsqueeze(0).to(device)
        
        inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
        inputs.to(device)
        
        img_f = model.get_image_features(**inputs)
        batch_img_f_list.append(img_f)
        
        if idx % batch_size == 0 or idx == len(image_paths):
            # Calculate the batch mean image feature
            batch_img_f_tensor = torch.stack(batch_img_f_list)
            batch_mean_img_f = torch.mean(batch_img_f_tensor, dim=0)
            # Convert to numpy array and save batch mean image feature to npz file
            np.savez(os.path.join(save_path, f'{foldername}_batch_{idx}_mean_img_f.npz'), features=batch_mean_img_f.cpu().numpy())
            
            # Clear the list for the next batch
            batch_img_f_list = []
            
        
# Load and calculate total mean_img_f for each class
# for foldername in class_num_dic.keys():
#     data = np.load(os.path.join(save_path, f'{foldername}_batch_img_f.npz'))
#     batch_img_f_tensor = torch.tensor(data['features']).to(device)
#     mean_img_f = torch.mean(batch_img_f_tensor, dim=0)
    
    # You can now use 'mean_img_f' for your calculations or storage
    # For example, you could save it to another npz file
    # np.savez(os.path.join(save_path, f'{foldername}_mean_img_f.npz'), features=mean_img_f.cpu().numpy())

# print(img_f_list[0])
#     stacked_img_f = torch.stack(img_f_list,dim = 0)
# print(stacked_img_f.shape)
# mean_img_f = torch.mean(stacked_img_f,dim = 0)
# print(mean_img_f.shape)



# path2 = "../datasets/synthesized_cifar100/ray/image_ray_text6_9.png"
# image2 = Image.open(path2)
# image2 = tf(image2) 
# image2 = image2.unsqueeze(0).to(device) 
# inputs2 =processor(images  = image2,return_tensors = 'pt')
# inputs2.to(device)
# img_f2 =model.get_image_features(**inputs2)
# print(img_f2.shape)

# cos_score = torch.cosine_similarity(mean_img_f,img_f2,dim = 1)
# print(cos_score)
# mul_score = torch.matmul(mean_img_f,img_f2.T)
# print(mul_score)
# euc_dis = torch.norm(mean_img_f-img_f2,p = 2,dim = 1)
# print(euc_dis)


# # visual
# plt.figure(figsize=(12, 6))

# # Histograms
# plt.subplot(1, 2, 1)
# plt.hist(mean_img_f.detach().cpu().numpy().flatten(), bins=50, alpha=0.5, label='mean_img_f')
# plt.hist(img_f2.detach().cpu().numpy().flatten(), bins=50, alpha=0.5, label='img_f2')
# plt.title('Histogram of Feature Vectors')
# plt.xlabel('Feature Values')
# plt.ylabel('Frequency')
# plt.legend()

# # Kernel Density Plots
# plt.subplot(1, 2, 2)
# sns.kdeplot(mean_img_f.detach().cpu().numpy().flatten(), label='mean_img_f')
# sns.kdeplot(img_f2.detach().cpu().numpy().flatten(), label='img_f2')
# plt.title('Kernel Density Plot of Feature Vectors')
# plt.xlabel('Feature Values')
# plt.ylabel('Density')
# plt.legend()

# plt.tight_layout()
# plt.show()

# output_path = 'feature_distribution_visualization.png'
# plt.savefig(output_path)


#ToDo: use Eucledian distance to check the distance between real set and synthesized set 
# 코사인 distacne image간의 distacne 의미...?