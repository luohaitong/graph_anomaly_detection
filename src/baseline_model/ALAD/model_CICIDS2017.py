import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sn


class Discriminatorxz(nn.Module):
    def __init__(self, z_dim=100, spectral_norm=False):
        super(Discriminatorxz, self).__init__()
        layer = sn if spectral_norm else nn
        # Inference over x
        self.nn1x = layer.Linear(3, 8)
        self.bn1x = layer.BatchNorm1d(8)

        # Inference over z
        self.nn1z = layer.Linear(z_dim, 8)

        # Joint inference
        self.nn1xz = layer.Linear(16, 8)
        self.nn2xz = layer.Linear(8, 1)

    def inf_x(self, x):
        x = self.bn1x(self.nn1x(x))
        x = F.leaky_relu(x, negative_slope=0.2)

        return x

    def inf_z(self, z):
        z = F.dropout(F.leaky_relu(self.nn1z(z), negative_slope=0.2), 0.2)

        return z

    def inf_xz(self, xz):
        intermediate = F.dropout(F.leaky_relu(self.nn1xz(xz), negative_slope=0.2), 0.2)
        xz = self.nn2xz(intermediate)
        return xz, intermediate

    def forward(self, x, z):
        x = self.inf_x(x)
        z = self.inf_z(z)
        xz = torch.cat((x, z), dim=1)
        out, intermediate, = self.inf_xz(xz)
        return out, intermediate


class Discriminatorxx(nn.Module):
    def __init__(self, spectral_norm=False):
        super(Discriminatorxx, self).__init__()
        layer = sn if spectral_norm else nn
        # Inference over x
        self.nn1xx = layer.Linear(6, 3)
        self.nn2xx = layer.Linear(3, 1)


    def forward(self, x, x_hat):
        xx = torch.cat((x, x_hat), dim=1)
        intermediate = F.dropout(F.leaky_relu(self.nn1xx(xx), negative_slope=0.2), 0.2)
        out = self.nn2xx(intermediate)
        return out, intermediate


class Discriminatorzz(nn.Module):
    def __init__(self, z_dim=100, spectral_norm=False):
        super(Discriminatorzz, self).__init__()
        layer = sn if spectral_norm else nn
        # Inference over x
        self.nn1zz = layer.Linear(2*z_dim, z_dim)
        self.nn2zz = layer.Linear(z_dim, 1)

    def forward(self, z, z_hat):
        zz = torch.cat((z, z_hat), dim=1)
        intermediate = F.dropout(F.leaky_relu(self.nn1zz(zz), negative_slope=0.2), 0.2)
        out = self.nn2zz(intermediate)

        return out, intermediate


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.nn1 = nn.Linear(z_dim, 3)

    def forward(self, z):
        z = F.relu(self.nn1(z))
        return z


class Encoder(nn.Module):
    def __init__(self, z_dim=100, spectral_norm=False):
        super(Encoder, self).__init__()
        layer = sn if spectral_norm else nn
        self.nn1 = layer.Linear(3, z_dim)

    def forward(self, x):
        x = F.leaky_relu(self.nn1(x), negative_slope=0.2)

        return x
