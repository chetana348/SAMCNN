import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
    

class DataGen(Dataset):
    def __init__(self, image_root, gt_root, size, mode='train'):
        self.mode = mode
        self.image_paths = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                                   if f.endswith('.tif') or f.endswith('.png')])
        self.label_paths = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)
                                   if f.endswith('.tif')])

        self.size = size

        
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()])
                                 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L')
        label = np.array(label)
        label = (label > 0).astype(np.uint8) * 255
        label= Image.fromarray(label)

        #print(label.max())

        # Apply transforms
        image = self.transform(image)
        label = self.gt_transform(label)

        if self.mode == 'train':
            return {'image': image, 'label': label}
        else:
            name = os.path.basename(self.image_paths[idx])
            return {'image': image, 'label': label, 'name': name}

        
