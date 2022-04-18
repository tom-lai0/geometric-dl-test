import matplotlib.pyplot as plt
from data import *
from train import *
from functions import *
from torch.utils.data import DataLoader
from model import VAE
from plot_save_fig import *

# import numpy as np
# import pandas as pd
# from matplotlib import cm
# import seaborn as sns
# import torch
# import random
# import plotly.express as px
# import plotly.graph_objects as go


def main(config):
    dg = DataGenerator()
    x = dg.generate(num_data=config['num_data'])
    original_data = x.data
    model = VAE()
    model.set_reparam(config['reparam'])
    dataloader = DataLoader(original_data, config['batch_size'])

    history = train(original_data, dataloader, model, config)

    encoded, _ = model.encode(original_data)
    encoded = encoded.cpu().detach().numpy()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 9))

    plot_encoded(f, (ax1, ax2, ax3), encoded, original_data, config['exp_path'] + 'encoded.png')
    plot_loss_hist(f, (ax1, ax2, ax3), history, config)
    plot_logvar_hist(history, config) if config['reparam'] else None

    _, _, _, reconstructed = model(original_data)
    reconstructed = reconstructed.cpu().detach().numpy()

    save_3d_project_fig(f, (ax1, ax2, ax3), reconstructed, original_data, 
        config['exp_path'] + 'reconstruction.png')
    save_3d_project_fig(f, (ax1, ax2, ax3), original_data, original_data, 
        config['exp_path'] + 'original.png')


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

config = {
    'num_data': 1024 * 4,
    'device': device,
    'n_epochs': 5000,
    'batch_size': 128 * 1,
    'reparam': False,
    'losses': [],
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.001,
        'betas': (0.9, 0.999)
    },
    'epoch_per_fig': 50,
    'verbose': True,
    'exp_path': 'experiments/experiment_4/',
    'save_path': 'save/save_4.pt'
}


if __name__ == '__main__':
    main(config)
