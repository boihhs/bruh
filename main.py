import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from loadData import loadData
from model import network
import torch
from loss import customLoss


# Get the input folder
folder = input('Input folder where the images are: ')

# See if the input folder exists
while os.path.isdir(folder) == False:
    print('Error: Please enter a correct folder path')
    folder = input('Input folder where the images are: ')

while True:
    imageIndex = input('Input the index of the image: ')
    try:
        # Attempt to convert the input to an integer
        index = int(imageIndex)
        break
    except ValueError:
        print("Error: That is not a valid integer. Please try again.")

while True:
    threshold = input('Input threshold for the model: ')
    try:
        # Attempt to convert the input to an integer
        threshold = float(threshold)
        break
    except ValueError:
        print("Error: That is not a valid float. Please try again.")

with torch.no_grad():
    data = loadData(folder)
    image, label = data[index]

    net = network()

    device = torch.device("cpu")
    net.load_state_dict(torch.load('weights.pth'))

    test = net(image.unsqueeze(0).to(device))

    # Generate Default Points
    numberOfPointsAlongHorizontal = 40
    default_points_x = torch.linspace(1/(2*numberOfPointsAlongHorizontal), 1-1/(2*numberOfPointsAlongHorizontal), numberOfPointsAlongHorizontal)

    numberOfPointsAlongVertical = 40
    default_points_y = torch.linspace(1/(2*numberOfPointsAlongVertical), 1-1/(2*numberOfPointsAlongVertical), numberOfPointsAlongVertical)

    default_points = torch.cartesian_prod(default_points_x, default_points_y)

    #offsets = test[0, :, 0:1]

    conf = torch.exp(test[0, :, 0])/(torch.exp(test[0, :, 0])+torch.exp(torch.ones(1,numberOfPointsAlongHorizontal*numberOfPointsAlongVertical).to(device)-test[0, :, 0]))

    # Normalize and prepare the image
    image = (image + 1) / 2
    image_np = image.permute(1, 2, 0).numpy()

    # Get dimensions for plotting labels
    height = image_np.shape[0]
    width = image_np.shape[1]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the unmodified image
    axes[0].imshow(image_np)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    # Plot the modified image with annotations
    axes[1].imshow(image_np)
    axes[1].axis('off')
    axes[1].set_title('Modified Image')

    # Add labels to the modified image
    for i in range(label.shape[0]):
        axes[1].scatter(label[i, 0] * width, 
                        label[i, 1] * height, 
                        color='red', 
                        marker='o', 
                        s=2)

    for i in range(default_points.shape[0]):
        if conf[0][i].item() > threshold:
            axes[1].scatter(
                default_points[i, 0] * width,
                default_points[i, 1] * height,
                color='black',
                marker='o',
                s=10,
                alpha=conf[0][i].item(),
            )

    # Display the plots
    plt.tight_layout()
    plt.show()
