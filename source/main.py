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

def plot_3d_scatter(d, dir):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        d[:, 0],
        d[:, 1],
        d[:, 2],
        c = d[:, dir]
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.xlim(-1.4, 1.4)
    plt.ylim(-1.4, 1.4)
    fig.show()

def save_3d_project_fig(d, path):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 9))
    ax1.scatter(d[:, 0], d[:, 1], c=d[:, 2], marker='.')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('z')

    ax2.scatter(d[:, 0], d[:, 2], c=d[:, 1], marker='.')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title('y')

    ax3.scatter(d[:, 1], d[:, 2], c=d[:, 0], marker='.')
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')
    ax3.set_title('x')
    f.savefig(path)

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

    l = len(history['exp_inv_dist'])

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 9))
    ax1.scatter(np.log(range(1, l+1)), history['exp_inv_dist'], marker='.')
    ax1.set_title('exp_inv_dist')
    ax2.scatter(np.log(range(1, l+1)), history['kl_loss'], marker='.')
    ax2.set_title('kl_loss')
    ax3.scatter(np.log(range(1, l+1)), history['recon_loss'], marker='.')
    ax3.set_title('recon_loss')
    plt.savefig(config['exp_path'] + 'loss_history.png')

    _, _, _, exp_res2 = model(d)
    exp_res2 = exp_res2.cpu().detach().numpy()

    save_3d_project_fig(exp_res2, config['exp_path'] + 'reconstruction.png')
    save_3d_project_fig(d, config['exp_path'] + 'original.png')


config = {
    'num_data': 1024,
    'device': 'cpu',
    'n_epochs': 5000,
    'batch_size': 128,
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.001,
        'betas': (0.9, 0.999)
    },
    'epoch_per_fig': 500,
    'verbose': True,
    'exp_path': 'experiment_2/',
    'save_path': 'save/save_2.pt'
}


if __name__ == '__main__':
    main(config)
