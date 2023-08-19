import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

class Discriminator(nn.Module):   #Normal Convolution
    def __init__(self, img_channels, dsc_features):
        super(Discriminator, self).__init__()
        # batch_size x channels x 64 x 64
        self.model = nn.Sequential(
            nn.Conv2d( #batch_size x channels x 32 x 32
                in_channels=img_channels,
                out_channels=dsc_features,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),
            self.convblock( #batch_size x channels x 16 x 16
                in_channels=dsc_features,
                out_channels=dsc_features*2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            self.convblock( #batch_size x channels x 8 x 8
                in_channels=dsc_features*2,
                out_channels=dsc_features*4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            self.convblock( #batch_size x channels x 4 x 4
                in_channels=dsc_features*4,
                out_channels=dsc_features*8,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Conv2d( #batch_size x channels x 1 x 1 single value representing fake or real
                in_channels=dsc_features*8,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0
            ),
            nn.Sigmoid() #Cast 1x1 to sigmoid for classification
        )

    def convblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):  #Transpose Convolution
    def __init__(self, noise_dim, img_channels, gen_features):
        super(Generator, self).__init__()
        # batch_size x noise_dim x 1 x 1
        self.model = nn.Sequential(
            self.Tconvblock( # batch_size x gen_features*16 x 4 x 4
                in_channels=noise_dim,
                out_channels=gen_features*16,
                kernel_size=4,
                stride=2,
                padding=0
            ),
            self.Tconvblock(  # batch_size x gen_features*16 x 8 x 8
                in_channels=gen_features*16,
                out_channels=gen_features * 8,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            self.Tconvblock(  # batch_size x gen_features*16 x 16 x 16
                in_channels=gen_features * 8,
                out_channels=gen_features * 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            self.Tconvblock(  # batch_size x gen_features*16 x 32 x 32
                in_channels=gen_features * 4,
                out_channels=gen_features * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ConvTranspose2d( # batch_size x gen_features*16 x 64 x 64
                in_channels=gen_features*2,
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def Tconvblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

def initialize_weights_normal(models):
    for model in models:
        for module in model.modules():
            if type(module)==nn.Conv2d or type(module)==nn.ConvTranspose2d or type(module)==nn.BatchNorm2d:
                nn.init.normal_(module.weight.data, 0.0, 0.2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 3e-4
batch_size = 128
img_size = 64
img_channels = 3
noise_dim = 100
epochs = 10
dsc_features = 64
gen_features = 64
img_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*img_channels, [0.5]*img_channels)
])

dataset = datasets.ImageFolder(root="celeb_dataset", transform=img_transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(
    noise_dim=noise_dim,
    img_channels=img_channels,
    gen_features=gen_features
).to(device)

discriminator = Discriminator(
    img_channels=img_channels,
    dsc_features=dsc_features
).to(device)

initialize_weights_normal([generator, discriminator])

gen_opt = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
dsc_opt = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

loss = nn.BCELoss()

constant_noise = torch.randn(32, noise_dim, 1, 1).to(device)
tns_real = SummaryWriter(f"logs/real")
tns_fake = SummaryWriter(f"logs/fake")
step = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

        #Adversarial Game Losses :
        #Discriminator tries to maximize BCE loss while Generator tries to minimize it
        #Discriminator = max log(D(xi))+log(1-D(G(z))) xi->real img vec, z->noise vec
        #Generator = max log(D(G(z)))

        fake = generator(noise)
        dsc_real = discriminator(real).reshape(-1)
        dsc_fake = discriminator(fake.detach()).reshape(-1)
        dsc_real_loss = loss(dsc_real, torch.ones_like(dsc_real)) #second term of BCELoss = 0
        dsc_fake_loss = loss(dsc_fake, torch.zeros_like(dsc_fake)) #first term of BCELoss = 0
        dsc_loss = (dsc_real_loss + dsc_fake_loss) / 2
        discriminator.zero_grad()
        dsc_loss.backward()
        dsc_opt.step()

        dsc_output = discriminator(fake).reshape(-1)
        gen_loss = loss(dsc_output, torch.ones_like(dsc_output))
        generator.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        if not batch_idx % 200:
            with torch.no_grad():
                fake = generator(constant_noise)
                tns_fake.add_image("Fake", utils.make_grid(fake, 8, normalize=True), global_step=step)
                tns_real.add_image("Real", utils.make_grid(real, 8, normalize=True), global_step=step)
            step+=1





