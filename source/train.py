import torch
from time import time
from functions import *
import matplotlib.pyplot as plt


def train(tr_set, model, config):

    start_time = time()

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
        print(f'Training: the {epoch + 1} epoch.')

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

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: ')
            print(loss_hist['exp_inv_dist'][-1])
            print(loss_hist['kl_loss'][-1])
            print(loss_hist['recon_loss'][-1])
            print('-' * 20)

            # plt.figure()
            # plt.scatter()


    torch.save(model.state_dict(), f='experiment\\save_1.pt')

    end_time = time()
    print(f'Training: Finished in {end_time - start_time}')

    return loss_hist
