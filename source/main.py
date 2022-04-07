from pyexpat import model
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

dg = DataGenerator()
x = dg.generate(num_data=1024)
d = x.data
model = VAE()

config = {
    'device': 'cpu',
    'n_epochs': 100,
    'batch_size': 128,
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.001,
        'betas': (0.9, 0.999)
    },
    'early_stop': 200,
    'save_path': 'models/model.pth'
}

dataloader = DataLoader(
    d, config['batch_size'],
)

history = train(dataloader, model, config)

exp_res, _ = model.encode(d)
exp_res = exp_res.cpu().detach().numpy()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 9))
ax1.scatter(exp_res[:, 0], exp_res[:, 1], c=d[:, 0])
ax1.set_title('x')
ax2.scatter(exp_res[:, 0], exp_res[:, 1], c=d[:, 1])
ax2.set_title('y')
ax3.scatter(exp_res[:, 0], exp_res[:, 1], c=d[:, 2])
ax3.set_title('z')
plt.savefig('experiment\\exp_1.png')

plt.figure()
plt.plot(np.log(history['exp_inv_dist']))
plt.plot(history['kl_loss'])
plt.plot(history['recon_loss'])
plt.savefig('experiment\\history.png')
