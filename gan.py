
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

from dcgan import DC_GAN
from aux_funcs import freeze_model, unfreeze_model, weight_init
from reconstruct import recon_mask


def obstructImage(image, height, width):
    # For now apply a box obstruction, but later apply a random boolean mask obstruction.
    # Actually only need the mask, box obstruction for visualisation
    # Then we can just use this mask for indexing for region replacement and weight adjustment
    image_height = image.shape[2]
    image_width = image.shape[3]
    border = 8
    ### Apply a random obstruction to the image (For now apply a grey box)
    top_left_y = np.random.randint(border, image_height-height-border)
    top_left_x = np.random.randint(border, image_width-width-border)
    bottom_right_y = top_left_y+image_height
    bottom_right_x = top_left_x+image_width
    obstructed_region = [top_left_y, bottom_right_y, top_left_x, bottom_right_x]              # y1, y2, x1, x2
    obstructed_image = torch.tensor(image)  # copy image
    obstructed_image[0, :, top_left_y:bottom_right_y, top_left_x: bottom_right_x] = ((96/255)-0.5)*2   # Normalise with [-1,1]
    return obstructed_image, obstructed_region





# def reconImage(obs_images, mask, generator, discriminator):
#
#     # TODO: MAKE THIS HANDLE MULTIPLE IMAGES (BACKPROPATE VIA A VECTOR OF LOSSES)
#
#     mse_weights = getMSEWeightsMaskless(image_size=64).to(device)
#
#     # ONLY WORKS WITH ONE IMAGE
#     generator.eval()
#     discriminator.eval()
#     # To speed up computations we turn off the autograd engine for the network parameters.
#     # Cannot use torch.no_grad(), since this will also prevent gradient calculation for total_loss
#     freeze_model(generator)
#     freeze_model(discriminator)
#     num_steps = 2000
#     obs_images = obs_images.to(device)
#
#     outobsvis = torchvision.utils.make_grid((obs_images*mask.float()).detach().cpu(), normalize=True)   # Not black since in [-1,1]
#     plt.imshow(outobsvis.numpy().transpose((1, 2, 0)))
#
#     num_images = obs_images.shape[0]
#     noise_dim = 100
#     BCE_R = nn.BCELoss()
#     MSE_R = nn.MSELoss()
#     label_real = torch.full((num_images,), 1, device=device)
#
#
#     # Need to ensure that requires_grad is set to True after the variable is created on the desired device
#     latent_noise = torch.randn(num_images, noise_dim, 1, 1, device=device, requires_grad=True)
#     #latent_noise = torch.randn(num_images, noise_dim, 1, 1, device=device, requires_grad=True)
#
#     lr = 0.2
#     reconOptim = optim.SGD([latent_noise], lr=lr)
#
#     recon_loss_scale = 10
#
#     gen_progress = []
#     losses = []
#     for step in range(num_steps):
#         ### THEN TRY WITH ADAM
#         gen_images = generator(latent_noise)
#         if step % 50 == 0:
#             gen_progress.append(gen_images.detach().cpu())
#         D_gen_out = discriminator(gen_images)
#         D_gen_loss = BCE_R(D_gen_out, label_real)
#
#         # Let's try MSE LOSS ONLY IN CENTRAL REGION
#         recon_loss = weightedMSELoss(obs_images, gen_images, mse_weights, mask)        # Make this a function of corrupted box
#         # Find that we need to scale the reconstruction penalty much more
#         # We may even want to make the recon_loss scale a function of the number of steps. We would want to initially
#         # have a highly penalised reconstruction loss to get close to the image we want to reconstruct.
#         # Then we we are closer we wish to make that image more realistic
#
#         total_loss = D_gen_loss + recon_loss_scale*recon_loss
#         losses.append(total_loss.item())
#
#         reconOptim.zero_grad()
#         total_loss.backward()
#         reconOptim.step()
#
#         # USING MANUAL UPDATE (NO OPTIMISER)
#         # gradient = latent_noise.grad.detach()
#         # latent_noise.data -= lr*gradient
#         #
#         # latent_noise.grad.zero_()
#     unfreeze_model(generator)
#     unfreeze_model(discriminator)
#     gen_progress = torch.cat(gen_progress, dim=0)
#     plt.figure(figsize=(12,8))
#     outvis = torchvision.utils.make_grid(gen_progress.detach().cpu(), normalize=True)
#     plt.imshow(outvis.numpy().transpose((1, 2, 0)))
#
#     recon_fill = obs_images*mask.float() + generator(latent_noise)*(~mask).float()
#     out_fill = torchvision.utils.make_grid(torch.cat((recon_fill, obs_images, obs_images*mask.float())).detach().cpu(), normalize=True)
#     plt.figure(figsize=(6,6))
#     plt.imshow(out_fill.numpy().transpose((1, 2, 0)))
#     plt.axis('off')
#     plt.show()
#     print('a')


if __name__ == "__main__":

    ### TESTING WITH REPRODUCIBILITY. Remove when training
    # torch.manual_seed(0)
    # np.random.seed(0)

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
    # Apply black box obstruction transform using a lambda function probably
    #trainsamp = datautils.SubsetRandomSampler(fixedsplit_train)    # Doesn't work with shuffle
    # If we want a fixed split use: datautils.Subset(dataset, indices), and initialise a dataloader with this
    num_train = round(0.95*len(dataset))
    num_test = len(dataset) - num_train
    traindata, testdata = datautils.random_split(dataset, [num_train, num_test])

    # Create the dataloaders
    trainloader = datautils.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    testloader = datautils.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=n_workers)


    noise_dim = 100
    # G = DC_Generator(64).to(device)
    # G.apply(weight_init)
    #
    # D = DC_Discriminator(64).to(device)
    # D.apply(weight_init)
    #
    #
    lr_nGAN = 0.0002
    lr_wGAN = 0.00005
    num_epochs = 20
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

    #
    # BCE_L = nn.BCELoss()
    # D_losses = []
    # G_losses = []
    #
    # fixed_latent_noise = torch.randn(8, noise_dim, 1, 1).to(device)
    # test_images_log = []
    #
    # model_name = "FirstModelCropped144"
    # Save_Path = os.path.join(Model_Path, model_name + ".pt")
    # start_epoch = 0

    mask = torch.ones((64, 64), dtype=torch.bool, device=device)
    mask[10:50, 20:50] = 0
    recon_mask(next(iter(testloader))[0][0:6], mask, nGAN.G, nGAN.D, device=device)

    # Apply basic Frechet distance calculation
    # inceptionModel = InceptionV3().to(device)
    # fid_value = figGenVsCelebA(G, inceptionModel, num_images=10000, batch_size=100, cuda=cuda)

    #test_images_log = [G(fixed_latent_noise)]
    #visualiseProgress(test_images_log)


    # Training
    # G.train()
    # D.train()
    # for epoch in range(start_epoch, num_epochs):
    #
    #     for it, data in enumerate(trainloader):
    #         images, _ = data
    #         images = images.to(device)
    #
    #         b_size = images.shape[0]
    #         label_real = torch.full((b_size,), 1, device=device)
    #         label_fake = torch.full((b_size,), 0, device=device)
    #
    #         ### TRAIN DISCRIMINATOR
    #         freeze_model(G)
    #         latent_noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
    #         G_out = G(latent_noise)
    #
    #         D_real_out = D(images)
    #         D_real_loss = BCE_L(D_real_out, label_real)
    #
    #         D_fake_out = D(G_out)
    #         D_fake_loss = BCE_L(D_fake_out, label_fake)
    #
    #         D_train_loss = (D_real_loss + D_fake_loss)/2        # One way of standardising disc and gen losses
    #         D.zero_grad()
    #         D_train_loss.backward()
    #         D_optimiser.step()
    #         D_losses.append(D_train_loss.item())
    #         unfreeze_model(G)
    #
    #         ### TRAIN GENERATOR
    #         freeze_model(D)
    #         latent_noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
    #         G_out = G(latent_noise)     # Should we use a new latent noise vector?
    #
    #         D_result = D(G_out)
    #         G_train_loss = BCE_L(D_result, label_real)    # Minimising -log(D(G(z)) instead of maximising -log(1-D(G(z))
    #
    #         G.zero_grad()
    #         G_train_loss.backward()
    #         G_optimiser.step()
    #         G_losses.append(G_train_loss.item())
    #         unfreeze_model(D)
    #
    #         #clear_output(True)
    #         print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
    #               % (epoch+1, num_epochs, it+1, len(trainloader), D_train_loss, G_train_loss))
    #
    #     # Log output
    #     test_fake = G(fixed_latent_noise)
    #     test_images_log.append(test_fake.detach())
    #
    #     visualiseProgress(test_images_log, "Fake_Images_Progress_Cropped"+str(epoch+1))
    #     # Maintain the most recent model state. Copy to disk
    #     torch.save({
    #         'epoch':                epoch,
    #         'gen_state_dict':       G.state_dict(),
    #         'disc_state_dict':      D.state_dict(),
    #         'gen_opt_state_dict':   G_optimiser.state_dict(),
    #         'disc_opt_state_dict':  D_optimiser.state_dict(),
    #         'gen_loss':             G_losses,
    #         'disc_loss':            D_losses,
    #     }, Save_Path)
    #
    # plotLosses(G_losses, D_losses, len(trainloader))

    plt.show()
