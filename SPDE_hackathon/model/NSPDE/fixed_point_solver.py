# Adapted from https://github.com/crispitagorico/torchspde
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#=============================================================================================
# Convolution in physical space = pointwise mutliplication of complex tensors in Fourier space
#=============================================================================================

def compl_mul2d(func_fft, kernel_tensor):
    return torch.einsum("bixt, ijxt -> bjxt", func_fft, kernel_tensor)


def compl_mul1d_time(func_fft, kernel_tensor):
    return torch.einsum("bixt, ijxt -> bjxt", func_fft, kernel_tensor)


def compl_mul3d(func_fft, kernel_tensor):
    return torch.einsum("bixyt, ijxyt -> bjxyt", func_fft, kernel_tensor)


def compl_mul2d_time(func_fft, kernel_tensor):
    return torch.einsum("bixyt, ijxyt -> bjxyt", func_fft, kernel_tensor)

#=============================================================================================
# Implementation of the inverse discrete Fourier transform to create the computational graph
# (x,t) -> z(x,t) 
#=============================================================================================

def inverseDFTn(u_ft, grid, dim, s=None): 
    # u_ft: (batch, channels, modesx, (possibly modesy), modest) 
    #    or (channels, channels, modesx, (possibly modesy), modest)
    # grid: (dim_x, (possibly dim_y), dim_t, d) d=len(dim)
    # or    (dim_x, (possibly dim_y), d) d=len(dim)
    # or    (dim_t, 1)
    # u: (batch, channels, modesx, (possibly modesy), dim_t) 
    # or (batch, channels, dim_x, (possibly dim_y), dim_t)

    assert len(grid.size()) == len(dim) + 1, 'Error grid size '
    if dim == [-1]:
        dim = [len(u_ft.size())-1]

    new_size = np.array(u_ft.shape)
    if s is not None:
        assert len(s)==len(dim), "'s' should have the same length as 'dim'" 
        new_size[dim] = s

    # pad the input on axis i where s[i]>u_ft.shape[i] for i in dimensions where the dft is computed
    padding = np.concatenate([ [0, 0] if i not in dim else [0, new_size[i]-u_ft.shape[i]] for i in range(len(u_ft.shape)-1, -1, -1) ])
    u_ft = F.pad(u_ft, tuple(padding))

    # reciprocal frequency grid 
    N = torch.tensor(grid.size()[:-1], device=grid.device)
    with torch.no_grad():  
        if len(grid.shape) == 2: 
            delta = grid[1,:] - grid[0,:]
            grid_freq = grid/(delta**2*N)
        if len(grid.shape) == 3: 
            delta = grid[1,1,:] - grid[0,0,:]
            grid_freq = grid/(delta**2*N)     
        elif len(grid.shape) == 4: 
            delta = grid[1,1,1,:] - grid[0,0,0,:]  
            grid_freq = grid/(delta**2*N)  
    
    shape = [u_ft.shape[i] if i in dim else 1 for i in range(len(u_ft.shape))]+[grid.shape[-1]]
    grid = grid.reshape(shape)
    grid_freq = grid_freq.reshape(shape)

    for i in range(len(dim)):

        u_ft_ = u_ft.unsqueeze(1 + dim[i])
        grid_ = grid.unsqueeze(dim[i])
        grid_freq_ = grid_freq.unsqueeze(1 + dim[i])
        
        grid_prod = grid_[..., i]*grid_freq_[..., i]    # grid_prod[k][j] = <x_j, s_k>  
    
        # basis functions
        basis = torch.exp(2.*np.pi*1j*grid_prod)  

        # compute inverse DFT
        u_ft = torch.sum(u_ft_*basis, axis=dim[i])/N[i] # devide by N[i] to match the ifft with norm='backward'

    return u_ft

#=============================================================================================
# Semigroup action is integration against a kernel
#=============================================================================================
class KernelConvolution(nn.Module):
    def __init__(self, channels, modes1, modes2, modes3=None):
        super(KernelConvolution, self).__init__()

        """ This module has a kernel parametrized as a complex tensor in the spectral domain. 
            The method forward computes S*H;
            The method forward_init computes S_t*u_0.
        """

        self.scale = 1. / (channels**2)

        # define kernel-tensor shape depending on dim of the problem
        if not modes3: # 1d
            self.modes = [modes1, modes2]
            self.dims = [2,3]
            self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, modes1, modes2,  dtype=torch.cfloat)) # K_theta in paper
        else: # 2d
            self.modes = [modes1, modes2, modes3]
            self.dims = [2,3,4]
            self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat)) # K_theta in paper
       
   
    def forward(self, z, grid=None, init=False):
        """ z: (batch, channels, dim_x, (possibly dim_y), dim_t)
        grid: dim_x, (possibly dim_y), dim_t, d) with d=2 or 3 """
        
        # lower and upper bounds of selected frequencies
        freqs = [ (z.size(2+i)//2 - self.modes[i]//2, z.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]
 
        if not init: # S * u

            # Compute FFT
            z_ft = torch.fft.fftn(z, dim=self.dims)
            z_ft = torch.fft.fftshift(z_ft, dim=self.dims)
 
            # Pointwise multiplication of kernel_tensor and func_fft
            out_ft = torch.zeros(z.size(), device=z.device, dtype=torch.cfloat)
            if len(self.modes)==2: # 1d case
                out_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] = compl_mul2d(z_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ], self.weights)
            else: # 2d case
                out_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1], freqs[2][0]:freqs[2][1] ] = compl_mul3d(z_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1], freqs[2][0]:freqs[2][1] ], self.weights)
            
            # Compute Inverse FFT  
            out_ft = torch.fft.ifftshift(out_ft, dim=self.dims) 
            
            # (*) if the grid is provided, then compute the final DFT_inverse by hand to make explicit the dependence on the input and allow for autograd to compute gradients.
            if grid is None:
                z = torch.fft.ifftn(out_ft, dim=self.dims)
            else:  
                z = inverseDFTn(out_ft, grid, self.dims)

            return z.real

        else: # S_t * z_0
            return self.forward_init(z, grid)

    
    def forward_init(self, z0_path, grid=None):
        """ z0_path: (batch, channels, dim_x, (possibly dim_y), dim_t)
            grid: dim_x, (possibly dim_y), dim_t, d) with d=2 or 3"""
        if grid is not None:
            gridx, gridt = grid[...,0,:-1], grid[...,-1].unsqueeze(-1)
            if len(self.modes)==2:
                gridt = gridt[0]
            else:
                gridt = gridt[0,0]

        # lower and upper bounds of selected frequencies
        freqs = [ (z0_path.size(2+i)//2 - self.modes[i]//2, z0_path.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)-1) ]

        # K_t = F_t^-1(K)  
        if grid is None: # (*)
            weights = torch.fft.ifftn(torch.fft.ifftshift(self.weights, dim=[-1]), dim=[-1], s=z0_path.size(-1))
        else:  
            weights = inverseDFTn(torch.fft.ifftshift(self.weights, dim=[-1]), gridt, dim=[-1], s=[z0_path.size(-1)])

        # Compute FFT of the input signal to convolve
        z_ft = torch.fft.fftn(z0_path, dim=self.dims[:-1])
        z_ft = torch.fft.fftshift(z_ft, dim=self.dims[:-1])

        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(z0_path.size(), device=z0_path.device, dtype=torch.cfloat)
        if len(self.modes)==2: # 1d case
            out_ft[:, :, freqs[0][0]:freqs[0][1], : ] = compl_mul1d_time(z_ft[:, :, freqs[0][0]:freqs[0][1] ], weights)
        else: # 2d case
            out_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1], : ] = compl_mul2d_time(z_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ], weights)


        # Compute Inverse FFT   
        out_ft = torch.fft.ifftshift(out_ft, dim=self.dims[:-1])

        if grid is None: # (*)
            z = torch.fft.ifftn(out_ft, dim=self.dims[:-1])
        else: 
            z = inverseDFTn(out_ft, gridx, self.dims[:-1])
        return z.real


#=============================================================================================
# SPDE solver: neural fixed point problem solved by Picard's iteration.
#=============================================================================================

class NeuralFixedPoint(nn.Module):
    def __init__(self, spde_func, n_iter, modes1, modes2, modes3=None):
        super(NeuralFixedPoint, self).__init__()

        # self.padding = int(2**(np.ceil(np.log2(abs(2*T-1)))))

        # number of Picard's iterations
        self.n_iter = n_iter
        
        # vector fields F and G
        self.spde_func = spde_func
        
        # semigroup
        self.convolution = KernelConvolution(spde_func.hidden_channels, modes1, modes2, modes3) 


    def forward(self, z0, xi, grid=None):
        """ - z0: (batch, hidden_channels, dim_x (possibly dim_y))
            - xi: (batch, forcing_channels, dim_x, (possibly dim_y), dim_t)
            - grid: (dim_x, (possibly dim_y), dim_t)
        """
        
        # if True 1d, else 2d
        assert len(xi.size()) in [4,5], '1d and 2d cases only are implemented '
        dim_flag = len(xi.size())==4

        # constant path
        if dim_flag:
            z0_path = z0.unsqueeze(-1).repeat(1, 1, 1, xi.size(-1)) 
        else:
            z0_path = z0.unsqueeze(-1).repeat(1, 1, 1, 1, xi.size(-1)) 

        # S_t * z_0
        z0_path =  self.convolution(z0_path, grid=grid, init=True) 

        # step 1 of Picard
        z = z0_path

        # Picard's iterations
        for i in range(self.n_iter):

            F_z, G_z = self.spde_func(z) 

            if dim_flag:
                G_z_xi = torch.einsum('abcde, acde -> abde', G_z, xi)
            else:
                G_z_xi = torch.einsum('abcdef, acdef -> abdef', G_z, xi)

            H_z_xi = F_z + G_z_xi

            if i==self.n_iter-1:
                y = z0_path + self.convolution(H_z_xi, grid=grid)
            else:
                y = z0_path + self.convolution(H_z_xi)
            
            z = y
        
        return y



# def inverseDFTn(u_ft, grid, dim, s=None): previous version of inverse dft, which did not scale.
#     # u_ft: (batch, channels, modesx, (possibly modesy), modest) 
#     #    or (channels, channels, modesx, (possibly modesy), modest)
#     # grid: (dim_x, (possibly dim_y), dim_t, d) d=len(dim)
#     # or    (dim_x, (possibly dim_y), d) d=len(dim)
#     # or    (dim_t, 1)
#     # u: (batch, channels, modesx, (possibly modesy), dim_t) 
#     # or (batch, channels, dim_x, (possibly dim_y), dim_t)

#     assert len(grid.size()) == len(dim) + 1, 'Error grid size '
#     if dim == [-1]:
#         dim = [len(u_ft.size())-1]

#     new_size = np.array(u_ft.shape)
#     if s is not None:
#         assert len(s)==len(dim), "'s' should have the same length as 'dim'" 
#         new_size[dim] = s

#     # pad the input on axis i where s[i]>u_ft.shape[i] for i in dimensions where the dft is computed
#     padding = np.concatenate([ [0, 0] if i not in dim else [0, new_size[i]-u_ft.shape[i]] for i in range(len(u_ft.shape)-1, -1, -1) ])
#     u_ft = F.pad(u_ft, tuple(padding))

#     # reciprocal frequency grid 
#     N = torch.tensor(grid.size()[:-1], device=grid.device)
#     with torch.no_grad():  
#         if len(grid.shape) == 2: 
#             delta = grid[1,:] - grid[0,:]
#             grid_freq = grid/(delta**2*N)
#         if len(grid.shape) == 3: 
#             delta = grid[1,1,:] - grid[0,0,:]
#             grid_freq = grid/(delta**2*N)     
#         elif len(grid.shape) == 4: 
#             delta = grid[1,1,1,:] - grid[0,0,0,:]  
#             grid_freq = grid/(delta**2*N)  
    
#     shape = [u_ft.shape[i] if i in dim else 1 for i in range(len(u_ft.shape))]+[grid.shape[-1]]
#     grid = grid.reshape(shape)
#     grid_freq = grid_freq.reshape(shape)
 
#     for i in range(len(dim)):
#         u_ft = u_ft.unsqueeze(len(dim) + dim[i])
#         grid = grid.unsqueeze(dim[i])
#         grid_freq = grid_freq.unsqueeze(len(dim) + dim[i])
        
#     grid_prod = torch.sum(grid*grid_freq, dim=-1)    # grid_prod[k][j] = <x_j, s_k>  
    
#     # basis functions
#     basis = torch.exp(2.*np.pi*1j*grid_prod)  

#     # compute inverse DFT
#     u = torch.sum(u_ft*basis, axis=dim)/torch.prod(N) # devide by N[i] to match the ifft with norm='backward'

#     return u