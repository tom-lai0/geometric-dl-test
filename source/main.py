import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import torch
import random
import plotly.express as px
import plotly.graph_objects as go
from data import *
from train import *
from functions import *
from torch.utils.data import DataLoader
from model import VAE
from plot_save_fig import *


def main(config):
    dg = DataGenerator()
    x = dg.generate(num_data=1024)
    d = x.data
    model = VAE()
    dataloader = DataLoader(
        d, config['batch_size'],
    )

    history = train(d, dataloader, model, config)

    exp_res, _ = model.encode(d)
    exp_res = exp_res.cpu().detach().numpy()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 9))

    ax1.scatter(exp_res[:, 0], exp_res[:, 1], c=d[:, 0], marker='.')
    ax1.set_title('x')
    ax2.scatter(exp_res[:, 0], exp_res[:, 1], c=d[:, 1], marker='.')
    ax2.set_title('y')
    ax3.scatter(exp_res[:, 0], exp_res[:, 1], c=d[:, 2], marker='.')
    ax3.set_title('z')
    f.savefig(config['exp_path'] + 'encoded.png')
    ax1.cla(); ax2.cla(); ax3.cla()

    l = len(history['exp_inv_dist'])
    iter_per_epoch = config['num_data'] / config['batch_size']

    ax1.scatter(np.log(range(1, l+1)) - np.log(iter_per_epoch), history['exp_inv_dist'], marker='.')
    ax1.set_title('exp_inv_dist')
    ax1.set_xlabel('log(number_of_epoch)')
    ax2.scatter(np.log(range(1, l+1)) - np.log(iter_per_epoch), history['kl_loss'], marker='.')
    ax2.set_title('kl_loss')
    ax2.set_xlabel('log(number_of_epoch)')
    ax3.scatter(np.log(range(1, l+1)) - np.log(iter_per_epoch), history['recon_loss'], marker='.')
    ax3.set_title('recon_loss')
    ax3.set_xlabel('log(number_of_epoch)')
    f.savefig(config['exp_path'] + 'loss_history.png')
    ax1.cla(); ax2.cla(); ax3.cla()

    _, _, _, exp_res2 = model(d)
    exp_res2 = exp_res2.cpu().detach().numpy()

    save_3d_project_fig(f, (ax1, ax2, ax3), exp_res2, d, config['exp_path'] + 'reconstruction.png')
    save_3d_project_fig(f, (ax1, ax2, ax3), d, d, config['exp_path'] + 'original.png')


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

config = {
    'num_data': 1024,
    'device': device,
    'n_epochs': 5000,
    'batch_size': 128,
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.001,
        'betas': (0.9, 0.999)
    },
    'epoch_per_fig': 1000,
    'verbose': True,
    'exp_path': 'experiment_3/',
    'save_path': 'save/save_3.pt'
}


if __name__ == '__main__':
    main(config)
