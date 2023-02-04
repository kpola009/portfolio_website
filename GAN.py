import torch
from torch import nn
from torch import optim
import torchvision
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import base64
from pathlib import Path

def transform_function(image):
    transform = transforms.Compose([transforms.Resize((128,128)),
                                    transforms.ToTensor()])
    real_img = transform(image)
    return(real_img)

def create_fake_image():
    torch.manual_seed(42)
    fake_image = torch.rand(1,100)

    return fake_image

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(49152, 10000),
            nn.ReLU(),
            nn.Linear(10000, 1),
            nn.Sigmoid())

    def forward(self, img):
        img = img.view(1, -1)
        out = self.linear(img)

        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(100, 10000),
            nn.LeakyReLU(),
            nn.Linear(10000, 4000),
            nn.LeakyReLU(),
            nn.Linear(4000, 49152))

    def forward(self, latent_space):
        latent_space = latent_space.view(1, -1)

        out = self.linear(latent_space)

        return out

discr = Discriminator()
gen = Generator()

opt_d = optim.SGD(discr.parameters(), lr=0.001, momentum=0.9)
opt_g = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)

criterion = nn.BCELoss()


discr_e = 4
gen_e = 3

def GAN(image, epochs):

    real_img = transform_function(image)
    fake_img = create_fake_image()

    for epoch in tqdm(range(epochs), total=epochs):
        for k in range(discr_e):
            opt_d.zero_grad()
            out_d1 = discr(real_img)
            loss_d1 = criterion(out_d1, torch.ones((1, 1)))
            loss_d1.backward()

            out_d2 = gen(fake_img).detach()
            loss_d2 = criterion(discr(out_d2), torch.zeros((1, 1)))
            loss_d2.backward()

            opt_d.step()

        for i in range(gen_e):
            opt_g.zero_grad()

            out_g = gen(fake_img)
            # loss_g =  criterion(discr(out_g.to(device)), torch.ones(1, 1).to(device))

            # ----Loss function in the GAN paper
            loss_g = 1.0 - (discr(out_g))
            loss_g.backward()

            opt_g.step()

    return out_g

# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded
#
# # Function to read and manipulate images
# def load_img(img):
#     im = Image.open(img)
#     image = np.array(im)
#     return image

# image = "images/Projects Content/GAN/truck.jpeg"
#
# image = load_img(image)
# image = Image.fromarray(np.uint8(image)).convert('RGB')
# output_image = GAN(image)
# plt.imshow(np.transpose(output_image.resize(3, 32, 32).detach().numpy(), (1, 2, 0)))
# plt.show()