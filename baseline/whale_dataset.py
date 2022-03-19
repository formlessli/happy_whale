# coding=utf-8
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import numpy as np
import os
import pandas as pd
from PIL import Image
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms 


def convert_label(data):
    columns=['species','individual_id']
    for f in columns:
        data[f]=data[f].map(dict(zip(data[f].unique(),range(0,data[f].nunique()))))
    return np.array(data).tolist()

#print(data)
class WhaleDataset(Dataset):
    '''
    数据集，存在3个特征
    image，piece, label
    '''
    def __init__(self, data_path):
        self.data_path = data_path
        data = pd.read_csv(os.path.join(self.data_path,'train.csv'))
        fn=convert_label(data)
        #fn = np.loadtxt(csv_path, dtype=np.unicode_ , delimiter=",",skiprows=1)
        self.data = []
        for line in fn:
            line = np.char.rsplit(np.char.strip(line), ' ')
            
            self.data.append(line)
        self.transform = transforms.Compose([                
                transforms.Resize(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
 #               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ])
        self.target_transform = None
        print('init final!')
        #print(self.data)
        
    def __getitem__(self, index):
        fn, piece, label = self.data[index]
        fn, piece, label = fn[0], np.array(int(piece[0])), np.array(int(label[0]))
        image = Image.open(os.path.join(self.data_path,'train_images',fn)).convert('RGB').resize((224, 224))
        #image.show()
        #image = np.array(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)    
            
        return {'image':image, 'piece':torch.from_numpy(piece), 'label':torch.from_numpy(label)}
    def __len__(self):
        return len(self.data)