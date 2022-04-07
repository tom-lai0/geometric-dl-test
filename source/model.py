import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder1 = nn.Linear(in_features=3, out_features=16)
        self.encoder2 = nn.Linear(in_features=16, out_features=16)
        self.encoder3_mu = nn.Linear(in_features=16, out_features=2)
        self.encoder3_sigma = nn.Linear(in_features=16, out_features=2)

        self.decoder1 = nn.Linear(in_features=2, out_features=16)
        self.decoder2 = nn.Linear(in_features=16, out_features=16)
        self.decoder3 = nn.Linear(in_features=16, out_features=3)

        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return mu, logvar, z, x

    def encode(self, x):
        x = F.leaky_relu(self.encoder1(x), 0.1)
        x = F.leaky_relu(self.encoder2(x), 0.1)
        mu = self.encoder3_mu(x)
        logvar = self.encoder3_sigma(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        z = torch.normal(mu, torch.exp(logvar))
        return z

    def decode(self, z):
        x = F.leaky_relu(self.decoder1(z), 0.1)
        x = F.leaky_relu(self.decoder2(x), 0.1)
        x = self.decoder3(x)
        return x


