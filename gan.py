
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as datautils
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import copy
import os
from inception import InceptionV3
from fid import figGenVsCelebA

from dcgan import DC_GAN, visualise_train_progress
from aux_funcs import freeze_model, unfreeze_model, weight_init
from reconstruct import recon_mask, backpropLatent, getReconWeightsMaskless

# Create function using meshgrid to create mask of arbitrary shape
def createMask(num_images, image_size, device, mask_pos=(20,25), mask_size=(10,10), rand_mask=False, rand_size=False):
    border = 8
    mask = torch.ones((num_images, image_size, image_size), dtype=torch.bool, device=device)
    for i in range(num_images):
        if rand_size:
            mask_size = np.random.randint(5, image_size - 2 * border, size=2)
        if rand_mask:
            # Apply a random obstruction to the image (For now apply a grey box)
            top_left_y = np.random.randint(border, image_size-mask_size[0]-border)
            top_left_x = np.random.randint(border, image_size-mask_size[1]-border)
            bottom_right_y = top_left_y+mask_size[0]
            bottom_right_x = top_left_x+mask_size[1]
        else:
            top_left_y = mask_pos[0]
            top_left_x = mask_pos[1]
            bottom_right_y = top_left_y+mask_size[0]
            bottom_right_x = top_left_x+mask_size[1]

        mask[i, top_left_y:bottom_right_y, top_left_x: bottom_right_x] = 0

    return mask

def getLowDiscVal(images, generator, discriminator, device):
    images = images.to(device)
    disc_vals = discriminator(images)
    disc_inds = torch.argsort(disc_vals)
    fig, axs = plt.subplots(1,4)
    fig.set_size_inches(16,5)

    latent_noise = torch.randn(64, 100, 1, 1, device=device, requires_grad=True)
    reconOptim = optim.Adam([latent_noise], lr=0.01)
    backpropLatent(images, torch.ones((64, 64), dtype=torch.bool, device=device), 1000, reconOptim, generator, discriminator, latent_noise, device)
    gen_images = generator(latent_noise)
    disc_gen = discriminator(gen_images)
    disc_gen_desc = torch.argsort(disc_gen, descending=True)
    low_disc_im = torch.cat((images[disc_inds[0:2]], gen_images[disc_gen_desc[0:2]]))
    low_disc_vals = torch.cat((disc_vals[disc_inds[0:2]], disc_gen[disc_gen_desc[0:2]]))

    for i in range(4):
        axs[i].imshow(0.5 * (low_disc_im[i].detach().cpu().numpy().transpose(1, 2, 0) + 1))
        axs[i].set_title('%.4f' % low_disc_vals[i].item())
        axs[i].axis('off')
    fig.tight_layout()

def pixelWeightChoices(filename, mask):
    n_arr = [0.5,1,4]
    fig, axs = plt.subplots(1,len(n_arr)+1)
    fig.set_size_inches(12,3)
    for i in range(len(n_arr)):
        pixWeights = getReconWeightsMaskless(64, n_arr[i])
        axs[i].imshow(pixWeights)
        axs[i].set_title('n = %.2f' % n_arr[i])
        #axs[i].axis('off')
    im_i = axs[len(n_arr)].imshow(getReconWeightsMaskless(64,1)*(mask.float().cpu()))
    axs[len(n_arr)].set_title('n = 1 with Mask')
    #plt.colorbar(im_i)
    fig.tight_layout()
    fig.savefig(filename)

def visualiseGAN(generator, filename):
    # Visualise model
    num_samples = 48
    plt.figure(figsize=(12,3))
    #test_noise = 2*torch.rand(48, 100, 1, 1, device=device)-10      # Uniform in [-1,1]     (Gives more 'nice faces', slighly less variety)
    test_noise = torch.randn(num_samples, 100, 1, 1, device=device)          # Normal(0,1)
    outvis = torchvision.utils.make_grid(generator(test_noise).detach().cpu(), normalize=True, nrow=12)
    plt.imshow(outvis.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == "__main__":

    ### TESTING WITH REPRODUCIBILITY. Remove when training
    torch.manual_seed(38)
    np.random.seed(38)
    Model_Path = "Models"
    if not os.path.isdir(Model_Path):
        os.makedirs(Model_Path)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    start_from_checkpoint = True
    train_from_checkpoint = True
    batch_size = 64
    n_workers = 4

    crop_image = True
    image_size = 64
    dataset = torchvision.datasets.ImageFolder(root="celeba",
                                               transform=transforms.Compose([
                                                   #transforms.Lambda(lambda img: transforms.functional.crop(img, ))
                                                   transforms.CenterCrop(144 if crop_image else image_size),
                                                   transforms.Resize(64),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Into [-1,1]
                                               ]))
    # If we want a fixed split use: datautils.Subset(dataset, indices), and initialise a dataloader with this
    num_train = round(0.95*len(dataset))
    num_test = len(dataset) - num_train
    traindata, testdata = datautils.random_split(dataset, [num_train, num_test])

    # Create the dataloaders
    trainloader = datautils.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    testloader = datautils.DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=n_workers)


    lr_nGAN = 0.0002
    lr_wGAN = 0.00005
    lr_wGAN_gp = 0.0001
    num_epochs = 30
    nGAN = DC_GAN(device, 'DC_GAN_64_bSize', gan_type = 'normal')
    nD_optimiser = optim.Adam(nGAN.D.parameters(), lr=lr_nGAN, betas=(0.5,0.999))
    nG_optimiser = optim.Adam(nGAN.G.parameters(), lr=lr_nGAN, betas=(0.5,0.999))
    nGAN.prep_train(nD_optimiser, nG_optimiser)
    if start_from_checkpoint:
        nGAN.load_checkpoint(train=train_from_checkpoint)

    wGAN = DC_GAN(device, 'wDC_GAN_64_bSize', gan_type = 'wgan')
    wD_optimiser = optim.RMSprop(wGAN.D.parameters(), lr=lr_wGAN)
    wG_optimiser = optim.RMSprop(wGAN.G.parameters(), lr=lr_wGAN)
    wGAN.prep_train(wD_optimiser, wG_optimiser)
    if start_from_checkpoint:
        wGAN.load_checkpoint(train=train_from_checkpoint)

    #wGAN.train_loop(num_epochs, trainloader)

    #nGAN.train_loop(num_epochs, trainloader)

    # visualiseGAN(wGAN.G, 'WGAN_faces')
    # visualiseGAN(nGAN.G, 'NGAN_faces')
    # wgan_ep_steps = [0,1,2,3,4,6,8,12,16,22]
    # visualise_train_progress([wGAN.test_images_log[i] for i in wgan_ep_steps], 'wGANProgress', wgan_ep_steps)

    num_test_images = 8
    test_iter = iter(testloader)
    mask = createMask(num_test_images, image_size, device, mask_pos=(24,24), mask_size=(20,20), rand_mask=True, rand_size=True)
    recon_mask(next(test_iter)[0][0:num_test_images], mask, wGAN.G, wGAN.D, device=device)

    # Apply basic Frechet distance calculation
    # inceptionModel = InceptionV3().to(device)
    # fid_value = figGenVsCelebA(wGAN.G, inceptionModel, num_images=20000, batch_size=100, cuda=cuda)
    # print(fid_value)

    #pixelWeightChoices('pixel_weight_powers', mask)

    plt.show()
