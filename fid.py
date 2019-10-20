
"""
Adapted from https://github.com/mseitzer/pytorch-fid
"""

import torch
import torch.utils.data as datautils
import torch.nn.functional as F
import numpy as np
from scipy import linalg

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
                                    dims=2048, noise_dim=100, cuda=False, verbose=False):
    # NEEDED, otherwise not enough CUDA memory available due to intermediate variables not being freed (required for grad)
    # (f(g(x))' = f'(g(x))*g'(x)
    device = torch.device("cuda" if cuda else "cpu")
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


def figGenVsCelebA(generator, inceptionModel, noise_dim=100, num_images=10000, batch_size=100, cuda=True):
    mu_gen, sigma_gen = calculate_activation_statistics(num_images, generator, inceptionModel, batch_size=batch_size,
                                        dims=2048, noise_dim=noise_dim, cuda=cuda, verbose=False)

    celeba_precalc = np.load('fid_stats_celeba_crop144_64_our_calc.npz')
    celeba_precalc_mu = celeba_precalc.f.mu
    celeba_precalc_sigma = celeba_precalc.f.sigma
    celeba_precalc.close()

    fid_value = calculate_frechet_distance(mu_gen, sigma_gen, celeba_precalc_mu, celeba_precalc_sigma, eps=1e-6)
    return fid_value


def fid_stats_data(dataset, inceptionModel, filename, n_workers, cuda):
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
