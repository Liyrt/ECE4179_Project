
import torch

import torch.nn as nn
import torch.nn.functional as F
import os
from aux_funcs import freeze_model, unfreeze_model, weight_init
import matplotlib.pyplot as plt
import torchvision


def visualise_train_progress(test_images_log, filename):
    num_its = len(test_images_log)
    all_ims = torch.cat(test_images_log,dim=0)
    plt.figure(figsize=(16, num_its*2))
    out = torchvision.utils.make_grid(all_ims.detach().cpu(), normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class DC_Generator(nn.Module):

    def __init__(self, ngf=64):
        super(DC_Generator, self).__init__()
        self.net = nn.Sequential(
            # Noise vector is 100D
            nn.ConvTranspose2d( 100, ngf * 8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),          # Inplace ReLU if possible,  saves memory
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64      # Color images
        )

    def forward(self, x):

        return self.net(x)


class DC_Discriminator(nn.Module):

    def __init__(self, ndf=64, gan_type = 'normal'):
        super(DC_Discriminator, self).__init__()

        self.layers = [            # No bias since batch norm includes bias
            nn.Conv2d(3, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=0, bias=False),
        ]
        if gan_type == 'normal':
            self.layers.append(nn.Sigmoid())        # Scale between 0 and 1

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):

        return self.net(x)


class DC_GAN():

    # A GAN object with its own training methods, and network structure

    def __init__(self, device, model_name, gan_type = 'normal', n_channel_scale=64):
        assert gan_type in ['normal', 'wgan', 'wgan_gp']        # DC gan types
        # WGAN with gradient penalty doesn't use batch normalisation
        self.model_name = model_name
        self.Model_Path = 'Models'
        self.gan_type = gan_type
        self.n_critic = 5
        self.device = device
        self.noise_dim = 100
        self.fixed_latent_noise = torch.randn(8, self.noise_dim, 1, 1).to(self.device)
        self.BCE_L = nn.BCELoss()
        self.G = DC_Generator(ngf=n_channel_scale).to(self.device)
        self.G.apply(weight_init)
        self.D = DC_Discriminator(ndf=n_channel_scale, gan_type = gan_type).to(self.device)
        self.D.apply(weight_init)
        self.gen_losses = []
        self.disc_losses = []
        self.start_epoch = 0
        pass


    def prep_train(self, G_optimiser, D_optimiser):
        # Call this before training the model
        self.G_optimiser = G_optimiser
        self.D_optimiser = D_optimiser


    def train_discriminator(self, b_size, images, label_real, label_fake):
        freeze_model(self.G)
        latent_noise = torch.randn(b_size, self.noise_dim, 1, 1, device=self.device)
        G_out = self.G(latent_noise)

        D_real_out = self.D(images)
        D_fake_out = self.D(G_out)

        if self.gan_type == 'normal':
            D_real_loss = self.BCE_L(D_real_out, label_real)
            D_fake_loss = self.BCE_L(D_fake_out, label_fake)
            D_train_loss = (D_real_loss + D_fake_loss) / 2  # One way of standardising disc and gen losses
            self.D.zero_grad()
            D_train_loss.backward()
            self.D_optimiser.step()
        elif self.gan_type == 'wgan':
            D_train_loss = -(torch.mean(D_real_out) - torch.mean(D_fake_out)) / 2  # One way of standardising disc and gen losses
            self.D.zero_grad()
            D_train_loss.backward()
            self.D_optimiser.step()
            # Clip weights within range
            for param in self.D.parameters():
                param.data.clamp(-0.01,0.01)        # Maybe replace with hyperparameter argument

        loss_val = D_train_loss.item()
        self.disc_losses.append(loss_val)

        unfreeze_model(self.G)
        return loss_val

    def train_generator(self, b_size, label_real):
        freeze_model(self.D)
        latent_noise = torch.randn(b_size, self.noise_dim, 1, 1, device=self.device)
        G_out = self.G(latent_noise)  # Should we use a new latent noise vector?

        D_result = self.D(G_out)

        if self.gan_type == 'normal':
            G_train_loss = self.BCE_L(D_result, label_real)  # Minimising -log(D(G(z)) instead of maximising -log(1-D(G(z))
            self.G.zero_grad()
            G_train_loss.backward()
            self.G_optimiser.step()
        elif self.gan_type == 'wgan':
            G_train_loss = -torch.mean(D_result)
            self.G.zero_grad()
            G_train_loss.backward()
            self.G_optimiser.step()

        loss_val = G_train_loss.item()
        self.gen_losses.append(loss_val)

        unfreeze_model(self.D)
        return loss_val


    def train_loop(self, num_epochs, trainloader):
        self.G.train()
        self.D.train()
        test_images_log = []
        for epoch in range(self.start_epoch, num_epochs):
            for it, data in enumerate(trainloader):
                images, _ = data
                images = images.to(self.device)

                b_size = images.shape[0]
                label_real = torch.full((b_size,), 1, device=self.device)
                label_fake = torch.full((b_size,), 0, device=self.device)

                ### TRAIN DISCRIMINATOR
                D_train_loss = self.train_discriminator(b_size, images, label_real, label_fake)

                ### TRAIN GENERATOR (every n_critic iterations, so that we train the generator on new batches always)
                if self.gan_type == 'wgan':
                    if it % self.n_critic == 0:
                        G_train_loss = self.train_generator(b_size, label_real)
                        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                              % (epoch + 1, num_epochs, it + 1, len(trainloader), D_train_loss, G_train_loss))
                    else:
                        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f'
                              % (epoch + 1, num_epochs, it + 1, len(trainloader), D_train_loss))
                else:
                    G_train_loss = self.train_generator(b_size, label_real)


                    print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                          % (epoch + 1, num_epochs, it + 1, len(trainloader), D_train_loss, G_train_loss))

                # clear_output(True)


            # Log output
            test_fake = self.G(self.fixed_latent_noise)
            test_images_log.append(test_fake.detach())

            visualise_train_progress(test_images_log, self.model_name + "_Progress_" + str(epoch + 1))
            # Maintain the most recent model state. Copy to disk
            torch.save({
                'epoch': epoch,
                'gen_state_dict': self.G.state_dict(),
                'disc_state_dict': self.D.state_dict(),
                'gen_opt_state_dict': self.G_optimiser.state_dict(),
                'disc_opt_state_dict': self.D_optimiser.state_dict(),
                'gen_loss': self.gen_losses,
                'disc_loss': self.disc_losses,
            }, os.path.join(self.Model_Path, self.model_name + ".pt"))
        print('Done Training!')


    def load_checkpoint(self, train=True):
        Load_Path = os.path.join(self.Model_Path, self.model_name + ".pt")
        if os.path.isfile(Load_Path):
            checkpoint = torch.load(Load_Path)
            self.G.load_state_dict(checkpoint['gen_state_dict'])
            self.D.load_state_dict(checkpoint['disc_state_dict'])
            if train:
                self.G_optimiser.load_state_dict(checkpoint['gen_opt_state_dict'])  # This also loads the previous lr
                self.D_optimiser.load_state_dict(checkpoint['disc_opt_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.gen_losses = checkpoint['gen_loss']
                self.disc_losses = checkpoint['disc_loss']
        else:
            raise ValueError("Checkpoint does not exist")

