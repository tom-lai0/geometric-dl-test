import matplotlib.pyplot as plt
import numpy as np


def plot_3d_scatter(d, dir):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(d[:, 0], d[:, 1], d[:, 2], c = d[:, dir])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.xlim(-1.4, 1.4)
    plt.ylim(-1.4, 1.4)
    fig.show()


def save_3d_project_fig(f, ax, d, dir, path):
    ax1, ax2, ax3 = ax
    ax1.cla(); ax2.cla(); ax3.cla()

    ax1.scatter(d[:, 0], d[:, 1], c=dir[:, 2], marker='.')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_title('z')
    ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-1.2, 1.2)

    ax2.scatter(d[:, 0], d[:, 2], c=dir[:, 1], marker='.')
    ax2.set_xlabel('x'); ax2.set_ylabel('z'); ax2.set_title('y')
    ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2)

    ax3.scatter(d[:, 1], d[:, 2], c=dir[:, 0], marker='.')
    ax3.set_xlabel('y'); ax3.set_ylabel('z'); ax3.set_title('x')
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2)

    f.savefig(path)
    ax1.cla(); ax2.cla(); ax3.cla()


def plot_encoded(f, ax, encoded, original, path):
    ax1, ax2, ax3 = ax

    ax1.scatter(encoded[:, 0], encoded[:, 1], c=original[:, 0], marker='.')
    ax1.set_title('x')

    ax2.scatter(encoded[:, 0], encoded[:, 1], c=original[:, 1], marker='.')
    ax2.set_title('y')

    ax3.scatter(encoded[:, 0], encoded[:, 1], c=original[:, 2], marker='.')
    ax3.set_title('z')

    f.savefig(path)
    ax1.cla(); ax2.cla(); ax3.cla()


def plot_loss_hist(f, ax, history, config):
    ax1, ax2, ax3 = ax

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


def plot_logvar_hist(history, config):
    f1 = plt.figure()
    plt.scatter(range(len(history['logvar_hist'])), history['logvar_hist'])
    f1.savefig(config['exp_path'] + 'logvar_hist.png')

