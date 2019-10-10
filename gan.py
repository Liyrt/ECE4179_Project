
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as datautils
import torchvision
import torchvision.transforms as transforms
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import copy
import os

class Generator(nn.Module):

    def __init__(self, ngf=64):
        super(Generator, self).__init__()
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


class Discriminator(nn.Module):

    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # No bias since batch norm includes bias
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
            nn.Sigmoid()        # Scale between 0 and 1
        )

    def forward(self, x):

        return self.net(x)


def weight_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

# Add function layer/modularity later
# def train_loop(num_epochs, generator, discriminator, trainloader, loss_fn, optimizer, loss_arr, model_name = 'unknown_model', start_from_checkpoint=True):
#     Save_Path_BestLoss = os.path.join(Base_Path, model_name + ".pt")
#     Save_Path_BestValAcc = os.path.join(Base_Path, model_name + "BestState.pt")
#     best_val_acc = 0
#     best_train_loss = np.inf
#     start_epoch = 0
#     if start_from_checkpoint:
#         start_epoch, loss_arr, acc_arr = loadCheckpoint(model, optimizer, model_name)
#
#     for epoch in range(start_epoch, num_epochs):
#         disc_loss, gen_loss = train_epoch(model, trainloader, loss_fn, optimizer)
#
#         #clear_output(True)
#
#         print('Epoch: ' + str(epoch) + ', Generator Loss: ' + str(disc_loss) + ', Discriminator Loss: ' + str(gen_loss))
#         loss_arr[:, epoch] = [disc_loss, gen_loss]
#
#         # Maintain the most recent model state (weights) (not necessarily the best training loss)
#         # COPY TO DISK
#         torch.save({
#             'epoch':                epoch,
#             'model_state_dict':     model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss':                 loss_arr,
#             'acc':                  acc_arr,
#         }, Save_Path_BestLoss)
#         # Maintain the model state (weights) that performs best on the validation set
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_state = copy.deepcopy(model.state_dict())            # COPY TO RAM
#             # COPY TO DISK
#             torch.save({
#                 'epoch':                epoch,
#                 'model_state_dict':     model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss':                 loss_arr,
#                 'acc':                  acc_arr,
#             }, Save_Path_BestValAcc)
#
#     return best_state


def loadCheckpoint(gen, gen_opt, disc, disc_opt, model_name):
    Load_Path = os.path.join(Model_Path, model_name + ".pt")
    if os.path.isfile(Load_Path):
        checkpoint = torch.load(Load_Path)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])    # This also loads the previous lr
        disc.load_state_dict(checkpoint['disc_state_dict'])
        disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        gen_loss = checkpoint['gen_loss']
        disc_loss = checkpoint['disc_loss']

        return start_epoch, gen_loss, disc_loss
    else:
        raise ValueError("Checkpoint does not exist")


def visualiseProgress(test_images_log, filename):
    num_its = len(test_images_log)
    all_ims = torch.cat(test_images_log,dim=0)
    plt.figure(figsize=(16, num_its*2))
    out = torchvision.utils.make_grid(all_ims.detach().cpu(), normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.savefig(filename)

def plotLosses(gen_losses, disc_losses, batch_per_ep):
    plt.figure(figsize=(16, 6))
    plt.plot(np.arange(0,len(gen_losses)), gen_losses, label='Generator')
    plt.plot(np.arange(0,len(disc_losses)), disc_losses, label='Discriminator')
    plt.legend(loc='best')
    plt.xlabel('Iteration (%.4f Batches Per Epoch)' % batch_per_ep)
    plt.ylabel('Loss')
    plt.tight_layout()


if __name__ == "__main__":
    Model_Path = "Models"
    if not os.path.isdir(Model_Path):
        os.makedirs(Model_Path)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    start_from_checkpoint = True

    batch_size = 500
    n_workers = 4
    image_size = 64
    dataset = torchvision.datasets.ImageFolder(root="celeba",
                                               transform=transforms.Compose([
                                                   transforms.Resize(image_size),
                                                   transforms.CenterCrop(image_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
    # To actually crop, remove resize, or change its argument

    #trainsamp = datautils.SubsetRandomSampler(fixedsplit_train)    # Doesn't work with shuffle
    # If we want a fixed split use: datautils.Subset(dataset, indices), and initialise a dataloader with this
    num_train = round(0.95*len(dataset))
    num_test = len(dataset) - num_train
    traindata, testdata = datautils.random_split(dataset, [num_train, num_test])

    # Create the dataloader
    trainloader = datautils.DataLoader(traindata, batch_size=batch_size,
                                             shuffle=True, num_workers=n_workers)
    testloader = datautils.DataLoader(testdata, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers)

    noise_dim = 100
    G = Generator(64).to(device)
    G.apply(weight_init)

    D = Discriminator(64).to(device)
    D.apply(weight_init)

    # real_batch = next(iter(testloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True).cpu(), (1, 2, 0)))

    lr = 0.0002
    num_epochs = 10
    D_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))
    G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))

    BCE_L = nn.BCELoss()
    D_losses = []
    G_losses = []

    fixed_latent_noise = torch.randn(8, noise_dim, 1, 1).to(device)
    test_images_log = []

    model_name = "FirstModel"
    Save_Path = os.path.join(Model_Path, model_name + ".pt")
    start_epoch = 0
    if start_from_checkpoint:
        start_epoch, G_losses, D_losses = loadCheckpoint(G, G_optimiser, D, D_optimiser, model_name)
    latent_noise = torch.randn(8, noise_dim, 1, 1, device=device)
    test_images_log = [G(latent_noise)]
    #visualiseProgress(test_images_log)
    for epoch in range(start_epoch, num_epochs):

        for it, data in enumerate(trainloader):
            images, _ = data
            images = images.to(device)

            b_size = images.shape[0]
            label_real = torch.full((b_size,), 1, device=device)
            label_fake = torch.full((b_size,), 0, device=device)

            ### TRAIN DISCRIMINATOR
            latent_noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            G_out = G(latent_noise)

            D_real_out = D(images)
            D_real_loss = BCE_L(D_real_out, label_real)

            D_fake_out = D(G_out)
            D_fake_loss = BCE_L(D_fake_out, label_fake)

            D_train_loss = (D_real_loss + D_fake_loss)/2        # One way of standardising disc and gen losses
            D.zero_grad()
            D_train_loss.backward()
            D_optimiser.step()
            D_losses.append(D_train_loss.item())

            ### TRAIN GENERATOR
            latent_noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            G_out = G(latent_noise)     # Should we use a new latent noise vector?

            D_result = D(G_out)
            G_train_loss = BCE_L(D_result, label_real)    # Minimising -log(D(G(z)) instead of maximising -log(1-D(G(z))

            G.zero_grad()
            G_train_loss.backward()
            G_optimiser.step()
            G_losses.append(G_train_loss.item())

            #clear_output(True)
            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                  % (epoch+1, num_epochs, it+1, len(trainloader), D_train_loss, G_train_loss))

        # Log output
        test_fake = G(fixed_latent_noise)
        test_images_log.append(test_fake.detach())

        visualiseProgress(test_images_log, "Fake_Images_Progress"+str(epoch+1))
        # Maintain the most recent model state (weights) (not necessarily the best training loss)
        # COPY TO DISK
        torch.save({
            'epoch':                epoch,
            'gen_state_dict':       G.state_dict(),
            'disc_state_dict':      D.state_dict(),
            'gen_opt_state_dict':   G_optimiser.state_dict(),
            'disc_opt_state_dict':  D_optimiser.state_dict(),
            'gen_loss':             G_losses,
            'disc_loss':            D_losses,
        }, Save_Path)

    plotLosses(G_losses, D_losses, len(trainloader))

    plt.show()
