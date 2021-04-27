import torch
import torch.nn as nn
import torch.nn.functional as F

# Base Class for Generator CNN
class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim        
        self.t_conv1 = nn.ConvTranspose2d(conv_dim, conv_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(conv_dim*8)
        self.t_conv2 = nn.ConvTranspose2d(conv_dim*8, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(conv_dim*4)
        self.t_conv3 = nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(conv_dim*2)
        self.t_conv4 = nn.ConvTranspose2d(conv_dim*2, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.fc = nn.Linear(z_size, conv_dim*4)
        print('z_size', z_size)
        

    def forward(self, x):
        batch_s = x.shape[0]
        x = self.fc(x)
        x = x.view(batch_s, self.conv_dim, 2, 2)        
        x = F.relu(self.batch_norm1(self.t_conv1(x)))
        x = F.relu(self.batch_norm2(self.t_conv2(x)))
        x = F.relu(self.batch_norm3(self.t_conv3(x)))
        x = self.t_conv4(x)
        x = F.tanh(x)
        return x