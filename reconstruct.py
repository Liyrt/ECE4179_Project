
import torch
import torch.nn as nn
from aux_funcs import freeze_model, unfreeze_model, weight_init
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision


def getMSEWeightsMaskless(image_size=64):
    xx, yy = np.meshgrid(range(image_size), range(image_size))
    xx_d = xx - image_size//2
    yy_d = yy - image_size//2
    mse_weights = xx_d**2+yy_d**2
    mse_weights = torch.from_numpy(np.power(1 - mse_weights/mse_weights.max(), 1/2))    # Square root scaling

    # index_tensor = torch.zeros((1,3,image_size,image_size))
    # for i in range(image_size):
    #     for j in range(image_size):
    #         index_tensor[0,:,i,j] = 1 - ((image_size//2 - i)**2 + (image_size//2 - j)**2)/((image_size**2)/2)

    return mse_weights.float()          # Very basic radial mask


def weightedMSELoss(image1, image2, mse_weights, mask, image_size = 64):
    # One weighting approach where the weights are directly related to distance from obstructed region
    mse_weights = mask.float() * mse_weights
    # TODO: Apply reweighting after masking
    # Maybe do l1 norm instead
    return torch.mean(mse_weights*(image1-image2)**2)


def backpropLatent(images, mask, num_steps, reconOptim, generator, discriminator, latent_noise, device):

    freeze_model(generator)
    freeze_model(discriminator)

    losses = []
    im_progress = []
    num_images = images.shape[0]
    label_real = torch.full((num_images,), 1, device=device)
    mse_weights = getMSEWeightsMaskless(image_size=64).to(device)

    recon_loss_scale = 10
    for step in range(num_steps):
        ### THEN TRY WITH ADAM
        gen_images = generator(latent_noise)
        if step % 50 == 0:
            im_progress.append(gen_images.detach().cpu())
        D_gen_out = discriminator(gen_images)
        D_gen_loss = F.binary_cross_entropy(D_gen_out, label_real)

        # Let's try MSE LOSS ONLY IN CENTRAL REGION
        recon_loss = weightedMSELoss(images, gen_images, mse_weights, mask)        # Make this a function of corrupted box
        # Find that we need to scale the reconstruction penalty much more
        # We may even want to make the recon_loss scale a function of the number of steps. We would want to initially
        # have a highly penalised reconstruction loss to get close to the image we want to reconstruct.
        # Then we we are closer we wish to make that image more realistic

        total_loss = D_gen_loss + recon_loss_scale*recon_loss
        losses.append(total_loss.item())

        reconOptim.zero_grad()
        total_loss.backward()
        reconOptim.step()

    unfreeze_model(generator)
    unfreeze_model(discriminator)

    return losses, im_progress


def recon_mask(images, mask, generator, discriminator, device):
    generator.eval()
    discriminator.eval()
    images = images.to(device)

    num_images = images.shape[0]
    noise_dim = 100
    latent_noise = torch.randn(num_images, noise_dim, 1, 1, device=device, requires_grad=True)
    lr_recon = 0.2
    reconOptim = optim.SGD([latent_noise], lr=lr_recon)
    num_steps = 2000
    losses, im_progress = backpropLatent(images, mask, num_steps, reconOptim, generator, discriminator, latent_noise, device)

    visualiseRecon(images, im_progress, latent_noise, mask, generator)
    print('a')

def visualiseRecon(images, im_progress_list, final_noise, mask, generator):
    num_images = images.shape[0]
    # Visualise progress of latent noise updates. Fix this!
    im_progress = torch.cat(im_progress_list, dim=0)
    plt.figure(figsize=(12,8))
    outvis = torchvision.utils.make_grid(im_progress.detach().cpu(),normalize=True)
    plt.imshow(outvis.numpy().transpose((1, 2, 0)))

    # Compare masked image, reconstructed, image and original image
    renormalise = lambda x: 0.5*(x+1)
    recon_np = renormalise(images*mask.float() + generator(final_noise)*(~mask).float()).detach().cpu().numpy().transpose(0,2,3,1)
    images_np = renormalise(images).detach().cpu().numpy().transpose(0,2,3,1)
    masked_np = (renormalise(images)*mask.float()).cpu().numpy().transpose(0,2,3,1)

    comp_images = [masked_np, recon_np, images_np]
    comp_fig, comp_axes = plt.subplots(num_images,3, squeeze=False)
    comp_fig.set_size_inches(6, num_images * 3)

    comp_titles = ['Masked', 'Reconstructed', 'Original']
    for i in range(3):
        comp_axes[0,i].set_title(comp_titles[i])

    for im_i in range(num_images):
        for j in range(3):
            comp_axes[im_i, j].imshow(comp_images[j][im_i])
            comp_axes[im_i, j].axis('off')

    plt.tight_layout()
    # recon_fill = images*mask.float() + generator(final_noise)*(~mask).float()
    # out_fill = torchvision.utils.make_grid(torch.cat((recon_fill, images, images*mask.float())).detach().cpu(), normalize=True)
