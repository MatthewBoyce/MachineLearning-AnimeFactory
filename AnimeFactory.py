from PIL import Image
import matplotlib.pylab as plt
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

# Import our CNN classes
from src.discriminator import Discriminator
from src.generator import Generator

# Load in data set
def get_data_loader(batch_size, image_size, data_dir='training-data/'):
    
    image_transforms = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),])
    indices = np.random.choice(63565, 50000) # get 50k random samples
    data_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir, transform=image_transforms), sampler=SubsetRandomSampler(indices), batch_size=batch_size)
    
    return data_loader

# Visualise Images
def imshow(image):  
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))

# model weights
def init_weights_normal(m):
    classname = m.__class__.__name__
    # apply initial weights to convolutional and linear layers
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if  hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0.0)

# construct gan      
def build_GAN(d_conv_dim, g_conv_dim, z_size):

    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(init_weights_normal)
    G.apply(init_weights_normal)

    print(D)
    print()
    print(G)
    
    return D, G

def scale(x, feature_range=(-1, 1)):
    return x*(feature_range[1] - feature_range[0]) + feature_range[0]

def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCELoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCELoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def train(D, G, n_epochs, print_every=50):
 
    if train_on_gpu:
        D.cuda()
        G.cuda()

    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=36
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(anime_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # train the discriminator on real and fake images
            if train_on_gpu:
                real_images = real_images.cuda()
            d_optimizer.zero_grad()
            D_real = D(real_images)
            d_real_loss = real_loss(D_real)
            z_flex = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z_flex = torch.from_numpy(z_flex).float()
            if train_on_gpu:
                z_flex = z_flex.cuda()

            fake_images = G(z_flex)
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # train the generator with an adversarial loss
            g_optimizer.zero_grad()
            z_flex = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z_flex = torch.from_numpy(z_flex).float()
            if train_on_gpu:
                z_flex = z_flex.cuda()
            fake_images = G(z_flex)
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake, True) # use real loss to flip labels
            g_loss.backward()
            g_optimizer.step()

            if batch_i % print_every == 0:
                losses.append((d_loss.item(), g_loss.item()))
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))


        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    return losses

# Save Images Form Epoc
def save_generates_images(epoch, samples):
    print(str(epoch))
    fig, axes = plt.subplots(figsize=(16,16), nrows=6, ncols=6, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((32,32,3)))
    plt.suptitle('Images generated by GAN (epoch = {})'.format(epoch+1), size=25)
    plt.savefig('sample_{}.png'.format(epoch+1))
    plt.close()

# Set Batch Size and Load Images
batch_size = 128 #512
image_size = 32
anime_train_loader = get_data_loader(batch_size, image_size)
images, _ = iter(anime_train_loader).next()

# show training data
fig = plt.figure(figsize=(20, 20))
plot_size=100
for idx in np.arange(plot_size):
    ax = fig.add_subplot(10, 10, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
plt.show()

# define model hyperparams
d_conv_dim = 32
g_conv_dim = 32
z_size = 100
D, G = build_GAN(d_conv_dim, g_conv_dim, z_size)
train_on_gpu = torch.cuda.is_available()
lr=0.0005
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.3, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.3, 0.999))

# Train
n_epochs = 10
losses = train(D, G, n_epochs=n_epochs)

# load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
for i in range(len(samples)):
    save_generates_images(i, samples)

    