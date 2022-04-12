import torch
from time import time
from functions import *
import matplotlib.pyplot as plt


def train(data, tr_set, model, config):
    
    start_time = time()
    verboseprint = print if config['verbose'] else lambda *a, **k: None
    verboseprint('Training Started')

    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    # optimizer = torch.optim.Adam(model.parameters())

    loss_hist = {
        'exp_inv_dist': [],
        'kl_loss': [],
        'recon_loss': []
    }
    epoch = 0

    while epoch < config['n_epochs']:
        model.train()

        for x in tr_set:
            optimizer.zero_grad()
            x.to(config['device'])

            mu_, logvar_, z_, x_ = model(x)

            loss1 = exp_inverse_dist_loss(z_, x, size=config['batch_size'], dim=(2, 3))
            loss2 = kl_loss(logvar_, mu_)
            loss3 = recon_loss(x_, x)
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            loss_hist['exp_inv_dist'].append(loss1.item())
            loss_hist['kl_loss'].append(loss2.item())
            loss_hist['recon_loss'].append(loss3.item())

        epoch += 1

        if epoch % config['epoch_per_fig'] == 0:
            verboseprint(f'Epoch {epoch}: ')
            verboseprint(loss_hist['exp_inv_dist'][-1])
            verboseprint(loss_hist['kl_loss'][-1])
            verboseprint(loss_hist['recon_loss'][-1])
            verboseprint('-' * 20)

            d, _ = model.encode(data)
            d = d.cpu().detach().numpy()

            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 9))
            ax1.scatter(d[:, 0], d[:, 1], c=data[:, 0], marker='.')
            ax1.set_title('x')
            ax2.scatter(d[:, 0], d[:, 1], c=data[:, 1], marker='.')
            ax2.set_title('y')
            ax3.scatter(d[:, 0], d[:, 1], c=data[:, 2], marker='.')
            ax3.set_title('z')
            plt.savefig(config['exp_path'] + f'encoded_epoch_{epoch}.png')


    torch.save(model.state_dict(), f=config['save_path'])

    end_time = time()
    print(f'Training Finished in {round(end_time - start_time, 2)}')

    return loss_hist
