import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from loadData import loadData
from model import network
import matplotlib.pyplot as plt
from loss import customLoss
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
from torch.utils.data import random_split

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])  # Stack images (all have the same size)
    labels = [item[1] for item in batch]  # Keep labels as a list of variable-length tensors
    return images, labels


if __name__ == '__main__':

    model = network()
    #model.init_weights()
    #model.load_state_dict(torch.load('weights.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("CUDA Available: ", torch.cuda.is_available())
    #print("Using GPU:", torch.cuda.get_device_name(0))

    data = loadData('WeedCrop.v1i.yolov5pytorch/train/')

    train_loader = DataLoader(data, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)

    criterion = customLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    num_epochs = 100

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Set model to training mode

        for images, labels in train_loader:
            # Move images and labels to the GPU
            images, labels = images.float().to(device), [label.float().to(device) for label in labels] 
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = 0
            for i in range(outputs.shape[0]):
                loss = loss + criterion(outputs[i, :], labels[i], device)
            
            if torch.isnan(loss).any():
                print(f"Loss became NaN at epoch {epoch}")
                break
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            # Get loss weighed by to total number if images in the batch
            running_loss += loss.item() * images.size(0) 
            
        # Get the average loss across the entire epoc
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), 'weights.pth')
    
    torch.save(model.state_dict(), 'weights.pth')