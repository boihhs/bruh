from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class loadData(Dataset):

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.images_path = os.path.join(self.folder_path, "images")
        self.labels_path = os.path.join(self.folder_path, "labels")
        self.transform = transforms.Compose([
                                            transforms.Resize((320, 320)),           # Resize to height 80, width 120
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                            transforms.ToTensor(),                  # Convert to tensor
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] for each channel
                                            ])

        self.images = [file for file in os.listdir(self.images_path) if file.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.images_path, self.images[idx])
        label_path = os.path.join(self.labels_path, os.path.splitext(self.images[idx])[0] + '.txt')

        image = Image.open(file_path).convert("RGB")
        

        data = np.loadtxt(label_path)
        data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 1:
            
            labels = data[1:3].unsqueeze(0)
        else:
            labels = data[:, 1:3]
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels