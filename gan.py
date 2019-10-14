
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
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import copy
import os
from inception import InceptionV3

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




def get_activations(images, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    model.eval()

    if images.shape[0] % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > images.shape[0]:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = images.shape[0]

    n_batches = images.shape[0] // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        torch_batch = images[start:end,:,:,:]

        # Reshape to (n_images, 3, height, width)
        # image_batch = image_batch.transpose((0, 3, 1, 2))
        # image_batch /= 255
        #
        # torch_batch = torch.from_numpy(image_batch).type(torch.FloatTensor)
        if cuda:
            torch_batch = torch_batch.cuda()
        with torch.no_grad():
            pred = model(torch_batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_activation_statistics(num_images, generator, inceptionModel, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    # NEEDED, otherwise not enough CUDA memory available due to intermediate variables not being freed (required for grad)
    # (f(g(x))' = f'(g(x))*g'(x)
    with torch.no_grad():
        n_batches = num_images // batch_size
        act = np.zeros((n_batches*batch_size, dims))
        # Do multiple image sub batches here. Save memory
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            latent_noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            gen_images = generator(latent_noise)
            # Normalise to [0,1] instead of [-1,1]. Inception then normalises again to [-1,1]
            gen_images += 1
            gen_images *= 0.5  # In-place python operations save GPU memory

            act[start:end] = get_activations(gen_images, inceptionModel, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    https://github.com/mseitzer/pytorch-fid
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def figGenVsCelebA(generator, inceptionModel, num_images=10000, batch_size=100, cuda=True):
    mu_gen, sigma_gen = calculate_activation_statistics(num_images, generator, inceptionModel, batch_size=batch_size,
                                        dims=2048, cuda=cuda, verbose=False)

    celeba_precalc = np.load('fid_stats_celeba.npz')
    celeba_precalc_mu = celeba_precalc.f.mu
    celeba_precalc_sigma = celeba_precalc.f.sigma
    celeba_precalc.close()

    fid_value = calculate_frechet_distance(mu_gen, sigma_gen, celeba_precalc_mu, celeba_precalc_sigma, eps=1e-6)
    return fid_value


def fid_stats_data(dataset, inceptionModel, filename):
    batch_size = 100
    full_loader_fid = datautils.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers, drop_last=True)

    act = np.zeros((batch_size*len(full_loader_fid), 2048))

    with torch.no_grad():
        for i, (images, samples) in enumerate(full_loader_fid):
            images += 1
            images *= 0.5  # In-place python operations save GPU memory

            act[i*batch_size:i*batch_size+batch_size] = get_activations(images, inceptionModel, batch_size, 2048, cuda=cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    np.savez(filename, mu=mu, sigma=sigma)
    return mu, sigma


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


def getMSEWeightsMaskless(image_size=64):
    xx, yy = np.meshgrid(range(image_size), range(image_size))
    xx_d = xx - image_size//2
    yy_d = yy - image_size//2
    mse_weights = xx_d**2+yy_d**2
    mse_weights = torch.from_numpy(1 - mse_weights/mse_weights.max())

    # index_tensor = torch.zeros((1,3,image_size,image_size))
    # for i in range(image_size):
    #     for j in range(image_size):
    #         index_tensor[0,:,i,j] = 1 - ((image_size//2 - i)**2 + (image_size//2 - j)**2)/((image_size**2)/2)

    return mse_weights.float()          # Very basic radial mask


def weightedMSELoss(image1, image2, mse_weights, mask, image_size = 64):
    # One weighting approach where the weights are directly related to distance from obstructed region
    mse_weights = mask.float() * mse_weights
    # TODO: Apply reweighting after masking

    return torch.mean(mse_weights*(image1-image2)**2)



def freeze_model(model):
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True



def reconImage(obs_images, mask, generator, discriminator):

    mse_weights = getMSEWeightsMaskless(image_size=64).to(device)

    # ONLY WORKS WITH ONE IMAGE
    generator.eval()
    discriminator.eval()
    # To speed up computations we turn off the autograd engine for the network parameters.
    # Cannot use torch.no_grad(), since this will also prevent gradient calculation for total_loss
    freeze_model(generator)
    freeze_model(discriminator)
    num_steps = 1000
    obs_images = obs_images.to(device)

    outobsvis = torchvision.utils.make_grid((obs_images*mask.float()).detach().cpu(), normalize=True)   # Not black since in [-1,1]
    plt.imshow(outobsvis.numpy().transpose((1, 2, 0)))

    num_images = obs_images.shape[0]
    noise_dim = 100
    BCE_R = nn.BCELoss()
    MSE_R = nn.MSELoss()
    label_real = torch.full((num_images,), 1, device=device)


    # Need to ensure that requires_grad is set to True after the variable is created on the desired device
    latent_noise = torch.randn(num_images, noise_dim, 1, 1, device=device, requires_grad=True)
    lr = 0.1
    reconOptim = optim.SGD([latent_noise], lr=lr)

    recon_loss_scale = 0

    gen_progress = []
    losses = []
    for step in range(num_steps):
        ### THEN TRY WITH ADAM
        gen_images = generator(latent_noise)
        if step % 50 == 0:
            gen_progress.append(gen_images.detach().cpu())
        D_gen_out = D(gen_images)
        D_gen_loss = BCE_R(D_gen_out, label_real)

        # Let's try MSE LOSS ONLY IN CENTRAL REGION
        recon_loss = weightedMSELoss(obs_images, gen_images, mse_weights, mask)        # Make this a function of corrupted box
        # Find that we need to scale the reconstruction penalty much more
        # We may even want to make the recon_loss scale a function of the number of steps. We would want to initially
        # have a highly penalised reconstruction loss to get close to the image we want to reconstruct.
        # Then we we are closer we wish to make that image more realistic

        total_loss = D_gen_loss + recon_loss_scale*recon_loss
        losses.append(total_loss.item())

        reconOptim.zero_grad()
        total_loss.backward()
        reconOptim.step()

        # USING MANUAL UPDATE (NO OPTIMISER)
        # gradient = latent_noise.grad.detach()
        # latent_noise.data -= lr*gradient
        #
        # latent_noise.grad.zero_()
    unfreeze_model(generator)
    unfreeze_model(discriminator)
    gen_progress = torch.cat(gen_progress, dim=0)
    plt.figure(figsize=(12,8))
    outvis = torchvision.utils.make_grid(gen_progress.detach().cpu(), normalize=True)
    plt.imshow(outvis.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":

    ### TESTING WITH REPRODUCIBILITY
    torch.manual_seed(0)
    np.random.seed(0)

    Model_Path = "Models"
    if not os.path.isdir(Model_Path):
        os.makedirs(Model_Path)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    start_from_checkpoint = True
    batch_size = 500
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

    # Create the dataloader
    trainloader = datautils.DataLoader(traindata, batch_size=batch_size,
                                             shuffle=True, num_workers=n_workers)
    testloader = datautils.DataLoader(testdata, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers)

    # inceptionModel = InceptionV3().to(device)


    # real_batch = next(iter(testloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True).cpu(), (1, 2, 0)))


    noise_dim = 100
    G = Generator(64).to(device)
    G.apply(weight_init)

    D = Discriminator(64).to(device)
    D.apply(weight_init)


    lr = 0.0002
    num_epochs = 10
    D_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))
    G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))

    BCE_L = nn.BCELoss()
    D_losses = []
    G_losses = []

    fixed_latent_noise = torch.randn(8, noise_dim, 1, 1).to(device)
    test_images_log = []

    model_name = "FirstModelCropped144"
    Save_Path = os.path.join(Model_Path, model_name + ".pt")
    start_epoch = 0
    if start_from_checkpoint:
        start_epoch, G_losses, D_losses = loadCheckpoint(G, G_optimiser, D, D_optimiser, model_name)

    mask = torch.ones((64, 64), dtype=torch.bool, device=device)
    mask[10:20, 20:30] = 0
    reconImage(dataset[0][0].unsqueeze(dim=0), mask, G, D)
    # Apply basic Frechet distance calculation

    # fid_value = figGenVsCelebA(G, inceptionModel, num_images=10000, batch_size=100, cuda=cuda)

    #test_images_log = [G(fixed_latent_noise)]
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

        visualiseProgress(test_images_log, "Fake_Images_Progress_Cropped"+str(epoch+1))
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
