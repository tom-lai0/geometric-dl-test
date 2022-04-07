import torch
from torch.nn import functional as F


def cal_dist(x, size = 16, dim = 3, device = 'cpu'):
    # return squared pairwise euclidean distance
    x.reshape((size, dim))
    x = torch.tensor(x, device = device)
    c = (
        (x ** 2).sum(axis = 1).reshape(size, 1) *
         torch.ones((size, size))
    )
    a = x @ x.T
    return c.T + c - 2 * a


def exp_inverse_dist_loss(
        output, origin, size=16, dim=(3, 3), device='cpu'
):
    output_mat = cal_dist(output, size, dim[0], device)
    origin_mat = cal_dist(origin, size, dim[1], device)

    loss = abs(
        torch.exp(-output_mat) - torch.exp(-origin_mat)
    ).sum().sum()

    return loss


def kl_loss(logvar, mu):
    return (
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    )


def recon_loss(recon_x, x):
    return F.cross_entropy(
        recon_x, x.float(), reduce='sum'
    )


