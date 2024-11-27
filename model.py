import torch
import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        # Initial Convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        # First Depthwise Separable Convolution Layer
        self.convDSC_1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bnDSC_1_1 = nn.BatchNorm2d(64)
        self.convDSC_1_2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.bnDSC_1_2 = nn.BatchNorm2d(32)

        # Second Depthwise Separable Convolution Layer
        self.convDSC_2_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.bnDSC_2_1 = nn.BatchNorm2d(32)
        self.convDSC_2_2 = nn.Conv2d(32, 64, kernel_size=1, padding=0)
        self.bnDSC_2_2 = nn.BatchNorm2d(64)

        # Third Depthwise Separable Convolution Layer
        self.convDSC_3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bnDSC_3_1 = nn.BatchNorm2d(64)
        self.convDSC_3_2 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.bnDSC_3_2 = nn.BatchNorm2d(128)

        # Forth Depthwise Separable Convolution Layer
        self.convDSC_4_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128)
        self.bnDSC_4_1 = nn.BatchNorm2d(128)
        self.convDSC_4_2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.bnDSC_4_2 = nn.BatchNorm2d(256)

        # Fith Depthwise Separable Convolution Layer
        self.convDSC_5_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256)
        self.bnDSC_5_1 = nn.BatchNorm2d(256)
        self.convDSC_5_2 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.bnDSC_5_2 = nn.BatchNorm2d(128)

        # Sixth Depthwise Separable Convolution Layer
        self.convDSC_6_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128)
        self.bnDSC_6_1 = nn.BatchNorm2d(128)
        self.convDSC_6_2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.bnDSC_6_2 = nn.BatchNorm2d(256)

        # Seventh Depthwise Separable Convolution Layer
        self.convDSC_7_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256)
        self.bnDSC_7_1 = nn.BatchNorm2d(256)
        self.convDSC_7_2 = nn.Conv2d(256, 512, kernel_size=1, padding=0)
        self.bnDSC_7_2 = nn.BatchNorm2d(512)

        #Final Convolution
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # Divides dimensions by 2

    # Function that passes input image(s) through VGG16 layers with ReLU activations
    def forward(self, x):
        #First Convolution
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        # First Depthwise Separable Convolution Layer
        x = self.convDSC_1_1(x)
        x = F.relu(self.bnDSC_1_1(x))
        x = self.convDSC_1_2(x)
        x = F.relu(self.bnDSC_1_2(x))
        x = self.maxpool(x)

        # Second Depthwise Separable Convolution Layer
        x = self.convDSC_2_1(x)
        x = F.relu(self.bnDSC_2_1(x))
        x = self.convDSC_2_2(x)
        x = F.relu(self.bnDSC_2_2(x))

        # Third Depthwise Separable Convolution Layer
        x = self.convDSC_3_1(x)
        x = F.relu(self.bnDSC_3_1(x))
        x = self.convDSC_3_2(x)
        x = F.relu(self.bnDSC_3_2(x))
        x = self.maxpool(x)

        # Forth Depthwise Separable Convolution Layer
        x = self.convDSC_4_1(x)
        x = F.relu(self.bnDSC_4_1(x))
        x = self.convDSC_4_2(x)
        x = F.relu(self.bnDSC_4_2(x))

        # Fith Depthwise Separable Convolution Layer
        x = self.convDSC_5_1(x)
        x = F.relu(self.bnDSC_5_1(x))
        x = self.convDSC_5_2(x)
        x = F.relu(self.bnDSC_5_2(x))

        # Sixth Depthwise Separable Convolution Layer
        x = self.convDSC_6_1(x)
        x = F.relu(self.bnDSC_6_1(x))
        x = self.convDSC_6_2(x)
        x = F.relu(self.bnDSC_6_2(x))

        # Seventh Depthwise Separable Convolution Layer
        x = self.convDSC_7_1(x)
        x = F.relu(self.bnDSC_7_1(x))
        x = self.convDSC_7_2(x)
        x = F.relu(self.bnDSC_7_2(x))
      
        # Final Convolution
        x = self.conv2(x)
        
        # Reshape Output
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1, x.shape[3])

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Initialize biases for Conv2d
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



        