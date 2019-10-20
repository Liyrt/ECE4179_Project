
import torch
import torch.nn as nn
from aux_funcs import freeze_model, unfreeze_model, weight_init
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision


def getReconWeightsMaskless(image_size=64,n=0.5):
    xx, yy = np.meshgrid(range(image_size), range(image_size))
    xx_d = xx - image_size//2
    yy_d = yy - image_size//2
    mse_weights = xx_d**2+yy_d**2           # IF with abs, then not Euclidean distance (not radial)
    mse_weights = torch.from_numpy(np.power(1 - mse_weights/mse_weights.max(), n))    # Square root scaling
    #mse_weights = torch.ones((image_size, image_size))
    # index_tensor = torch.zeros((1,3,image_size,image_size))
    # for i in range(image_size):
    #     for j in range(image_size):
    #         index_tensor[0,:,i,j] = 1 - ((image_size//2 - i)**2 + (image_size//2 - j)**2)/((image_size**2)/2)

    return mse_weights.float()          # Very basic radial mask


def weightedReconLoss(image1, image2, mse_weights, mask, image_size = 64):
    # One weighting approach where the weights are directly related to distance from obstructed region
    mask = mask.unsqueeze(1).expand_as(image1)
    mse_weights = mse_weights.unsqueeze(0).unsqueeze(0).expand_as(image1)
    mse_weights = mask.float() * mse_weights
    mse_weights /= (torch.max(mse_weights.view(image1.shape[0],-1), dim=1)[0]).view(image1.shape[0],1,1,1).expand_as(image1)
    # Maybe do l1 norm instead (give very similar results)
    return torch.mean(mse_weights*(image1-image2)**2, dim=tuple(range(1,len(image1.shape))))


def backpropLatent(images, mask, num_steps, reconOptim, generator, discriminator, latent_noise, device, scheduler=None):

    freeze_model(generator)
    freeze_model(discriminator)

    num_spoints = 15
    num_images = images.shape[0]
    losses = np.zeros((num_images, num_steps))
    im_progress = []
    label_real = torch.full((num_images,), 1, device=device)
    # Backpropagate with each image weighted the same amount. Mean reduction over the loss vector could have been
    # applied and the same effect (up to a factor = num_images) would have occurred. This is because the ith component
    # of the loss vector is only dependent on the ith original image and latent noise 'sub-tensor'
    backprop_weights = torch.ones((num_images,), device=device)
    mse_weights = getReconWeightsMaskless(image_size=64).to(device)

    recon_loss_scale = 20
    for step in range(num_steps):
        gen_images = generator(latent_noise)
        if step % (num_steps//num_spoints) == 0:
            im_progress.append(gen_images.detach().cpu())

        D_gen_out = discriminator(gen_images)

        # If critic output is not a probability (i.e. WGAN with weight clipping)
        # This is still not a great solution as the critic does not have to be centred at 0
        if not isinstance(list(discriminator.modules())[-1], torch.nn.Sigmoid):
            D_gen_out = torch.sigmoid(D_gen_out)            # May want to scale input so gradients flow better (less saturation)
            #D_gen_loss = torch.clamp(label_real - D_gen_out,-50,50)       # This is not a good idea

        # Losses for each image are separate (Backpropagate separately)
        D_gen_loss = F.binary_cross_entropy(D_gen_out, label_real, reduction='none')

        # Let's try MSE LOSS ONLY IN CENTRAL REGION  . Perhaps add an additional proximity loss for continuity
        recon_loss = weightedReconLoss(images, gen_images, mse_weights, mask)        # Make this a function of corrupted box
        # Find that we need to scale the reconstruction penalty much more
        # We may even want to make the recon_loss scale a function of the number of steps. We would want to initially
        # have a highly penalised reconstruction loss to get close to the image we want to reconstruct.
        # Then we we are closer we wish to make that image more realistic
        #recon_loss_scale = 20 - 20*np.clip(2*step/num_steps,0,1)
        total_loss = D_gen_loss + recon_loss_scale*recon_loss
        losses[:,step] = total_loss.detach().cpu().numpy()

        reconOptim.zero_grad()
        total_loss.backward(backprop_weights)
        reconOptim.step()

        if scheduler is not None:
            scheduler.step(total_loss.mean())
    # Perhaps check what the final discriminator loss and final reconstruction losses are.

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
    lr_recon = 0.05    # This needs to be adaptive somehow, or the optimiser must be quite stable against lr
    reconOptim = optim.Adam([latent_noise], lr=lr_recon)                ### TRY WITH ADAM (seems better)
    reconOptSched = optim.lr_scheduler.ReduceLROnPlateau(reconOptim, factor=0.2, patience=200)
    num_steps = 5000
    losses, im_progress = backpropLatent(images, mask, num_steps, reconOptim, generator, discriminator, latent_noise, device, reconOptSched)
    visualiseRecon(images, im_progress, latent_noise, mask, generator)
    plt.figure(figsize=(12, 4))
    plt.plot(losses.transpose((1,0)))
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.tight_layout()
    plt.legend(['Image' + str(i) for i in range(num_images)], loc='best')


def visualiseRecon(images, im_progress_list, final_noise, mask, generator):
    num_images = images.shape[0]
    # Visualise progress of latent noise updates. Fix this!
    im_progress = torch.cat(im_progress_list, dim=0)
    plt.figure(figsize=(6,8))
    outvis = torchvision.utils.make_grid(im_progress.detach().cpu(), nrow=num_images, normalize=True)
    plt.imshow(outvis.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.tight_layout()

    # Compare masked image, reconstructed, image and original image
    mask = mask.unsqueeze(1).expand_as(images)
    renormalise = lambda x: 0.5*(x+1)
    generated_np = generator(final_noise).detach().cpu()
    recon_np = renormalise(images*mask.float() + generator(final_noise)*(~mask).float()).detach().cpu()#.numpy().transpose(0,2,3,1)
    images_np = renormalise(images).detach().cpu()#.numpy().transpose(0,2,3,1)
    masked_np = (renormalise(images)*mask.float()).cpu()#.numpy().transpose(0,2,3,1)

    comp_images = [masked_np, images_np, recon_np, generated_np]
    comp_fig, comp_axes = plt.subplots(1,len(comp_images), squeeze=False)
    comp_fig.set_size_inches(5, num_images * 1.5)
    comp_fig.subplots_adjust(wspace=0, left=0.03, right=0.97, bottom=0.02, top=0.95)
    comp_titles = ['Masked', 'Original', 'Reconstructed', 'Generated']
    for i in range(len(comp_images)):
        comp_axes[0,i].set_title(comp_titles[i],fontsize=12)

    for j in range(len(comp_images)):
        current_vis = torchvision.utils.make_grid(comp_images[j].detach().cpu(), normalize=True, nrow=1)
        comp_axes[0, j].imshow(current_vis.numpy().transpose((1, 2, 0)))
        comp_axes[0, j].axis('off')
