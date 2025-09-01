import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator Network
    """
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # nc:  Number of channels in the input image (e.g., 3 for RGB)
        # ndf: Size of feature maps in discriminator (e.g., 64)

        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            # Conv2d is used to downsample the image
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Final output: a single probability value
        )

    def forward(self, input):
        """
        Defines the forward pass of the discriminator.
        """
        return self.main(input)