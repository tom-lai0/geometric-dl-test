import torch
from time import time
from functions import *
import matplotlib.pyplot as plt
from plot_save_fig import *


def train(original_data, tr_set, model, config):
    
    start_time = time()
    verboseprint = print if config['verbose'] else lambda *a, **k: None
    verboseprint('Training Started')

    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    # optimizer = torch.optim.Adam(model.parameters())

    loss_hist = {
        'exp_inv_dist': [],
        'kl_loss': [],
        'recon_loss': [],
        'logvar_hist': []
    }
    epoch = 0
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 9))

    while epoch < config['n_epochs']:
        model.train()

        for x in tr_set:
            optimizer.zero_grad()
            x.to(config['device'])

            mu_, logvar_, z_, x_ = model(x)

            loss1 = exp_inverse_dist_loss(mu_, x, size=config['batch_size'], dim=(2, 3))
            # loss2 = kl_loss(logvar_, mu_)
            loss2 = loss_1(mu_)
            loss3 = recon_loss_l1(x_, x)
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            loss_hist['exp_inv_dist'].append(loss1.item())
            loss_hist['kl_loss'].append(loss2.item())
            loss_hist['recon_loss'].append(loss3.item())

            if config['reparam']:
                loss_hist['logvar_hist'].append(torch.mean(logvar_).item())

        epoch += 1

        if epoch % config['epoch_per_fig'] == 0:
            verboseprint(f'Epoch {epoch}: ')
            verboseprint(loss_hist['exp_inv_dist'][-1])
            verboseprint(loss_hist['kl_loss'][-1])
            verboseprint(loss_hist['recon_loss'][-1])
            verboseprint('-' * 20)

            midway_encoded, _ = model.encode(original_data)
            midway_encoded = midway_encoded.cpu().detach().numpy()
            plot_encoded(f, (ax1, ax2, ax3), midway_encoded, original_data, 
                config['exp_path'] + f'encoded_epoch_{epoch}.png')
           
            _, _, _, recon = model(original_data)
            recon = recon.cpu().detach().numpy()
            save_3d_project_fig(f, (ax1, ax2, ax3), recon, original_data, 
                config['exp_path'] + f'recon_epoch_{epoch}.png')
            
            ax1.cla(); ax2.cla(); ax3.cla()

    torch.save(model.state_dict(), f=config['save_path'])

    end_time = time()
    verboseprint(f'Training Finished in {round(end_time - start_time, 2)}')

    return loss_hist
