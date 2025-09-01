import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator Network
    """
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        # nz:  Size of the latent z vector (e.g., 100)
        # ngf: Size of feature maps in generator (e.g., 64)
        # nc:  Number of channels in the output image (e.g., 3 for RGB)

        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            # ConvTranspose2d is used to upsample the image
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (nc) x 64 x 64
        )

    def forward(self, input):
        """
        Defines the forward pass of the generator.
        """
        return self.main(input)