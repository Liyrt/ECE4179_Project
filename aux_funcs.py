
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def freeze_model(model):
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True

def weight_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()


def plotLosses(gen_losses, disc_losses, batch_per_ep):
    plt.figure(figsize=(16, 6))
    plt.plot(np.arange(0,len(gen_losses)), gen_losses, label='Generator')
    plt.plot(np.arange(0,len(disc_losses)), disc_losses, label='Discriminator')
    plt.legend(loc='best')
    plt.xlabel('Iteration (%.4f Batches Per Epoch)' % batch_per_ep)
    plt.ylabel('Loss')
    plt.tight_layout()
