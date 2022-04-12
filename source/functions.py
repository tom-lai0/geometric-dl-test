import torch
from torch.nn import functional as F


def cal_dist(x, size = 16, dim = 3, device = 'cpu'):
    # return squared pairwise euclidean distance
    x.reshape((size, dim))
    c = (
        (x ** 2).sum(axis = 1).reshape(size, 1) *
        torch.ones((size, size))
    )
    a = x @ x.T
    return c.T + c - 2 * a

def cal_dist_1(x, size = 16, dim = 3, device = 'cpu'):
    # return squared pairwise euclidean distance
    x = torch.reshape(x, (size, dim)).clone().detach().float().requires_grad_()
    c = torch.mul(
        torch.reshape(torch.sum(torch.pow(x, 2), 1), (size, 1)),
        torch.ones((size, size))
    )
    a = torch.matmul(x, torch.transpose(x, 0, 1))
    return torch.transpose(c, 0, 1) + c - 2 * a


def exp_inverse_dist_loss(
        output, origin, size=16, dim=(3, 3), device='cpu'
):
    output_mat = cal_dist(output, size, dim[0], device)
    origin_mat = cal_dist(origin, size, dim[1], device)

    loss = F.mse_loss(
        torch.exp(-output_mat), torch.exp(-origin_mat),
        reduction='sum'
    )

    return loss


def kl_loss(logvar, mu):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def recon_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')


