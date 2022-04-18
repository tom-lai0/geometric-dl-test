import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder1 = nn.Linear(in_features=3, out_features=16)
        self.encoder2 = nn.Linear(in_features=16, out_features=16)
        # self.encoder3 = nn.Linear(in_features=16, out_features=16)
        # self.encoder4 = nn.Linear(in_features=16, out_features=16)
        self.encoder_mu = nn.Linear(in_features=16, out_features=2)
        self.encoder_sigma = nn.Linear(in_features=16, out_features=2)

        self.decoder1 = nn.Linear(in_features=2, out_features=16)
        self.decoder2 = nn.Linear(in_features=16, out_features=16)
        # self.decoder3 = nn.Linear(in_features=16, out_features=16)
        # self.decoder4 = nn.Linear(in_features=16, out_features=16)
        self.decoder_out = nn.Linear(in_features=16, out_features=3)

        self.reparam = True

    def set_reparam(self, is_reparam):
        self.reparam = is_reparam

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar) if self.reparam else mu
        x = self.decode(z)

        return mu, logvar, z, x

    def encode(self, x):
        # x = F.leaky_relu(self.encoder1(x), 0.2)
        # x = F.leaky_relu(self.encoder2(x), 0.2)
        # x = F.leaky_relu(self.encoder3(x), 0.2)
        # x = F.leaky_relu(self.encoder4(x), 0.2)

        x = F.relu(self.encoder1(x))
        x = x.pow(2)
        x = F.relu(self.encoder2(x))
        x = x.pow(2)
        mu = self.encoder_mu(x)

        logvar = self.encoder_sigma(x) if self.reparam else None

        return mu, logvar

    def reparametrize(self, mu, logvar):
        s = torch.normal(0, torch.ones_like(logvar)) * torch.exp(logvar)
        z = mu + s
        return z

    def decode(self, z):
        # x = F.leaky_relu(self.decoder1(z), 0.2)
        # x = F.leaky_relu(self.decoder2(x), 0.2)
        # x = F.leaky_relu(self.decoder3(x), 0.2)
        # x = F.leaky_relu(self.decoder4(x), 0.2)

        x = F.relu(self.decoder1(z))
        x = x.pow(2)
        x = F.relu(self.decoder2(x))
        x = x.pow(2)
        x = self.decoder_out(x)

        return x


