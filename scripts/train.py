import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

# Go up one level to import the models from the 'models' directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.generator import Generator
from models.discriminator import Discriminator

## Configuration & Hyperparameters
# ---------------------------------------------------------------------------

# Root directory for dataset.
# IMPORTANT: Make sure this path is correct for your machine.
dataroot = "C:/Users/saigo/Desktop/ProGAN-Project/data/celeba_hq_256"

# Number of worker threads for loading the data with the DataLoader
workers = 2
# Batch size during training.
batch_size = 128
# Spatial size of training images. All images will be resized to this size.
image_size = 64
# Number of channels in the training images. For color images this is 3.
nc = 3
# Size of the latent z vector (i.e., size of generator input).
nz = 100
# Size of feature maps in the generator.
ngf = 64
# Size of feature maps in the discriminator.
ndf = 64
# Number of training epochs.
num_epochs = 25
# Learning rate for optimizers.
lr = 0.0002
# Beta1 hyperparameter for Adam optimizers.
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

## Data Loading & Preparation
# ---------------------------------------------------------------------------

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


## The Main Training Execution Block
# ---------------------------------------------------------------------------

# This 'if' statement is the fix for the multiprocessing error on Windows
if __name__ == '__main__':
    ## Model Initialization
    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(nc, ndf).to(device)

    ## Loss Function and Optimizers
    criterion = nn.BCELoss()

    # Create a batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    ## The Training Loop
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated with previous gradients
            errD_fake.backward()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 100 == 0:
                print(f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}]  Loss_D: {errD.item():.4f}  Loss_G: {errG.item():.4f}')

        # After each epoch, save a grid of generated images to see progress
        with torch.no_grad():
            fake_images = netG(fixed_noise).detach().cpu()
        
        # Create a directory to save images if it doesn't exist
        if not os.path.exists('./generated_images'):
            os.makedirs('./generated_images')
            
        vutils.save_image(fake_images, f'./generated_images/epoch_{epoch+1}.png', normalize=True)

    print("--- Training Finished ---")