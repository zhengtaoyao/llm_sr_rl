# Adapted from https://github.com/crispitagorico/torchspde
#  (originally from https://github.com/zongyi-li/fourier_neural_operator)
# Modified for current implementation by the authors of SPDEBench

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model.utilities import *

import operator
from functools import reduce
from functools import partial


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last

        self.conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w = nn.Conv3d(width, width, 1)
        # self.bn = torch.nn.BatchNorm2d(width)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y, dim_t)"""

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)

        return x


class FNO_space2D_time(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, L, T):
        super(FNO_space2D_time, self).__init__()

        """
        The overall network. It contains L layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. L layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: a driving function observed at T timesteps + 3 locations (u(1, x, y), ..., u(T, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, t=T, c=T+3)
        output: the solution at T timesteps
        output shape: (batchsize, x=64, t=T, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.L = L
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(T + 3, self.width)
        # input channel is T+3: the solution of the first T timesteps + 3 locations (u(1, x, y), ..., u(T, x, y),  x, y, t)

        self.net = [FNO_layer(modes1, modes2, modes3, width) for i in range(self.L - 1)]
        self.net += [FNO_layer(modes1, modes2, modes3, width, last=True)]
        self.net = nn.Sequential(*self.net)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """ - x: (batch, dim_x, dim_y, T_out, T)
        """
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        # x1 = self.conv0(x)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

        x = self.net(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


# ===========================================================================
# Data Loaders
# ===========================================================================

def dataloader_fno_2d_xi(u, xi, ntrain=1000, ntest=200, T=51, sub_x=128, sub_t=1, batch_size=20, dataset=None):
    if dataset == 'sns':
        T, sub_t, sub_x = 100, 1, 4

    dim_x = u.size(1) // sub_x
    dim_t = T // sub_t

    u_train = u[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t]
    xi_train = xi[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t].reshape(ntrain, dim_x, dim_x, 1, dim_t).repeat(
        [1, 1, 1, dim_t, 1])

    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]
    xi_test = xi[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t].reshape(ntest, dim_x, dim_x, 1, dim_t).repeat(
        [1, 1, 1, dim_t, 1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_train, u_train),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               persistent_workers=True,
                                               drop_last=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_test, u_test),
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              persistent_workers=True,
                                              drop_last=False,
                                              num_workers=4)

    return train_loader, test_loader


def dataloader_fno_2d_u0(u, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=4, batch_size=20, dataset=None):
    if dataset == 'sns':
        T, sub_t, sub_x = 100, 1, 4

    u_train = u[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t]
    u0_train = u[:ntrain, ::sub_x, ::sub_x, 0].unsqueeze(-1).unsqueeze(-1)
    u0_train = u0_train.repeat([1, 1, 1, T // sub_t, T // sub_t])

    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]
    u0_test = u[-ntest:, ::sub_x, ::sub_x, 0].unsqueeze(-1).unsqueeze(-1)
    u0_test = u0_test.repeat([1, 1, 1, T // sub_t, T // sub_t])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, u_train), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, u_test), batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


# ===========================================================================
# Training functionalities
# ===========================================================================

def train_fno_2d(model, train_loader, test_loader, device, myloss, batch_size=20, epochs=5000,
                       learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20,
                       plateau_patience=None, plateau_terminate=None, delta=0,
                       checkpoint_file='checkpoint.pt'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    if plateau_patience is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=plateau_patience, threshold=1e-6, min_lr=1e-7)
    if plateau_terminate is not None:
        early_stopping = EarlyStopping(patience=plateau_terminate, verbose=False, delta=delta, path=checkpoint_file)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    losses_train = []
    losses_test = []

    try:

        for ep in range(epochs):

            model.train()

            train_loss = 0.
            for xi_, u_ in train_loader:
                loss = 0.
                xi_ = xi_.to(device)
                u_ = u_.to(device)
                optimizer.zero_grad()

                u_pred = model(xi_)
                u_pred = u_pred[..., 0]
                loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            test_loss = 0.
            with torch.no_grad():
                for xi_, u_ in test_loader:
                    loss = 0.

                    xi_ = xi_.to(device)
                    u_ = u_.to(device)

                    u_pred = model(xi_)
                    u_pred = u_pred[..., 0]
                    loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                    test_loss += loss.item()

            if plateau_patience is None:
                scheduler.step()
            else:
                scheduler.step(test_loss / ntest)
            if plateau_terminate is not None:
                early_stopping(test_loss / ntest, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if ep % print_every == 0:
                losses_train.append(train_loss / ntrain)
                losses_test.append(test_loss / ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Val Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        return model, losses_train, losses_test

    except KeyboardInterrupt:

        return model, losses_train, losses_test



def eval_fno_2d(model, test_dl, myloss, batch_size, device):
    ntest = len(test_dl.dataset)
    test_loss = 0.
    with torch.no_grad():
        for xi_, u_ in test_dl:
            loss = 0.
            xi_, u_ = xi_.to(device), u_.to(device)
            u_pred = model(xi_)
            u_pred = u_pred[..., 0]
            loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))
            test_loss += loss.item()
    # print('Loss: {:.6f}'.format(test_loss / ntest))
    return test_loss / ntest
