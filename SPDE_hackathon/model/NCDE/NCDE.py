# Adapted from https://github.com/crispitagorico/torchspde
#  (originally from https://github.com/patrick-kidger/NeuralCDE)

import torch
import torchcde
import csv
import itertools
from model.utilities import *

#===============================================================================================================
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
#===============================================================================================================
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="linear", solver='euler'):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels-1, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation
        self.solver = solver

    def forward(self, u0, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        # X0 = X.evaluate(X.interval[0])
        z0 = self.initial(u0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              # t = X.interval,
                              method=self.solver,
                              t=X._t)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        # z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


#===========================================================================
# Data Loaders
#===========================================================================

def dataloader_ncde_1d(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, normalizer=True, interpolation='linear', dataset=None):

    if dataset=='phi41':
        T, sub_t = 51, 1
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5

    u_train = u[:ntrain, :dim_x, 0:T:sub_t].permute(0, 2, 1)
    xi_train = xi[:ntrain, :dim_x, 0:T:sub_t].permute(0, 2, 1)
    
    t = torch.linspace(0., xi_train.shape[1], xi_train.shape[1])[None, :, None].repeat(ntrain, 1, 1)
    xi_train = torch.cat([t, xi_train], dim=2)

    u_test = u[-ntest:, :dim_x, 0:T:sub_t].permute(0, 2, 1)
    xi_test = xi[-ntest:, :dim_x, 0:T:sub_t].permute(0, 2, 1)

    t = torch.linspace(0., xi_test.shape[1], xi_test.shape[1])[None, :, None].repeat(ntest, 1, 1)
    xi_test = torch.cat([t,xi_test], dim=2)

    if normalizer:
        xi_normalizer = UnitGaussianNormalizer(xi_train)
        xi_train = xi_normalizer.encode(xi_train)
        xi_test = xi_normalizer.encode(xi_test)

        u_normalizer = UnitGaussianNormalizer(u_train)
        u_train = u_normalizer.encode(u_train)
        u_test = u_normalizer.encode(u_test)

    u0_train = u_train[:, 0, :]
    u0_test = u_test[:, 0, :]

    # interpolation
    if interpolation=='linear':
        xi_train = torchcde.linear_interpolation_coeffs(xi_train)
        xi_test = torchcde.linear_interpolation_coeffs(xi_test)
    elif interpolation=='cubic':
        xi_train = torchcde.hermite_cubic_coefficients_with_backward_differences(xi_train)
        xi_test = torchcde.hermite_cubic_coefficients_with_backward_differences(xi_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, u_normalizer


def dataloader_ncde_2d(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=1, batch_size=20, normalizer=True, interpolation='linear', dataset=None):

    if dataset=='sns':
        T, sub_t, sub_x = 100, 1, 4
    dim_x = u.size(1)//sub_x

    u0_train = u[:ntrain, ::sub_x, ::sub_x, 0].reshape(ntrain, dim_x*dim_x)
    u_train = u[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t].reshape(ntrain, dim_x*dim_x, -1).permute(0,2,1)
    xi_train = xi[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t].reshape(ntrain, dim_x*dim_x, -1).permute(0,2,1)

    t = torch.linspace(0., T, T)[None,:,None].repeat(ntrain,1,1)
    xi_train = torch.cat([t,xi_train], dim=2)

    u0_test = u[-ntest:, ::sub_x, ::sub_x, 0].reshape(ntest, dim_x*dim_x)
    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t].reshape(ntest, dim_x*dim_x, -1).permute(0,2,1)
    xi_test = xi[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t].reshape(ntest, dim_x*dim_x, -1).permute(0,2,1)

    if normalizer:
        xi_normalizer = UnitGaussianNormalizer(xi_train)
        xi_train = xi_normalizer.encode(xi_train)
        xi_test = xi_normalizer.encode(xi_test)

        u_normalizer = UnitGaussianNormalizer(u_train)
        u_train = u_normalizer.encode(u_train)
        u_test = u_normalizer.encode(u_test)

    u0_train = u_train[:, 0, :]
    u0_test = u_test[:, 0, :]

    # interpolation
    if interpolation=='linear':
        xi_train = torchcde.linear_interpolation_coeffs(xi_train)
        xi_test = torchcde.linear_interpolation_coeffs(xi_test)
    elif interpolation=='cubic':
        xi_train = torchcde.hermite_cubic_coefficients_with_backward_differences(xi_train)
        xi_test = torchcde.hermite_cubic_coefficients_with_backward_differences(xi_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, u_normalizer

#===========================================================================
# Training and testing functionalities
#===========================================================================
def eval_ncde(model, test_dl, myloss, batch_size, device, u_normalizer=None):

    ntest = len(test_dl.dataset)
    test_loss = 0.
    with torch.no_grad():
        for u0_, xi_, u_ in test_dl:    
            loss = 0.       
            u0_, xi_, u_ = u0_.to(device), xi_.to(device), u_.to(device)
            u_pred = model(u0_, xi_)

            if u_normalizer is not None:
                u_pred = u_normalizer.decode(u_pred.cpu())
                u_ = u_normalizer.decode(u_.cpu())

            loss = myloss(u_pred[:, 1:, :].reshape(batch_size, -1), u_[:, 1:, :].reshape(batch_size, -1))
            test_loss += loss.item()
    # print('Test Loss: {:.6f}'.format(test_loss / ntest))
    return test_loss / ntest

def train_ncde(model, train_loader, test_loader, u_normalizer, device, myloss, batch_size=20, epochs=5000, learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20, plateau_patience=None, delta=0, plateau_terminate=None, checkpoint_file='checkpoint.pt'):

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
            for u0_, xi_, u_ in train_loader:

                loss = 0.
                
                u0_ = u0_.to(device)
                xi_ = xi_.to(device)
                u_ = u_.to(device)

                u_pred = model(u0_, xi_)
                
                if u_normalizer is not None:
                    u_pred = u_normalizer.decode(u_pred.cpu())
                    u_ = u_normalizer.decode(u_.cpu())
                
                loss = myloss(u_pred[:, 1:, :].reshape(batch_size, -1), u_[:, 1:, :].reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            test_loss = 0.
            with torch.no_grad():
                for u0_, xi_, u_ in test_loader:
                    
                    loss = 0.
                    
                    u0_ = u0_.to(device)
                    xi_ = xi_.to(device)
                    u_ = u_.to(device)

                    u_pred = model(u0_, xi_)

                    if u_normalizer is not None:
                        u_pred = u_normalizer.decode(u_pred.cpu())
                        u_ = u_normalizer.decode(u_.cpu())

                    loss = myloss(u_pred[:, 1:, :].reshape(batch_size, -1), u_[:, 1:, :].reshape(batch_size, -1))

                    test_loss += loss.item()
            
            if plateau_patience is None:
                scheduler.step()
            else:
                scheduler.step(test_loss/ntest)
            if plateau_terminate is not None:
                early_stopping(test_loss/ntest, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Val Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        return model, losses_train, losses_test

    except KeyboardInterrupt:

        return model, losses_train, losses_test



