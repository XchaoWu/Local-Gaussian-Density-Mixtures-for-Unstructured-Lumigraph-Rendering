import numpy as np
import torch 
import torch.nn as nn 
import math 
from easydict import EasyDict as edict
import sys 
from HashGrid import HashGrid2D, PyHashGrid
import torch.nn.functional as F 
# from DoubleVector import VecorEncoding
from cuda import padding_inputs_forward, padding_inputs_backward

class Gaussian_Act(nn.Module):
    def __init__(self, sigma=0.1):
        super(Gaussian_Act, self).__init__()
        self.item = 1./ (-2 * (sigma ** 2))
    def forward(self, x):
        return torch.exp( (x ** 2) * self.item)

class GaussianAct(nn.Module):
    def __init__(self):
        super(GaussianAct, self).__init__()
    def forward(self, x, sigma):
        item = 1./ (-2 * (sigma ** 2))
        return torch.exp( (x ** 2) * item)

class Positional_Encoding(nn.Module):
    def __init__(self, L):
        super(Positional_Encoding, self).__init__()
        self.L = L 
    def embed(self, x, L):
        rets = [x]
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.**i*x))
        return torch.cat(rets, -1)   
    def forward(self, x):
        return self.embed(x, self.L)

    
class Weighted_Positional_Encoding(nn.Module):
    def __init__(self, L):
        super(Weighted_Positional_Encoding, self).__init__()
        self.L = L 
    def embed(self, x, L):
        rets = [x]
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.**i*x))
        return torch.cat(rets, -1)   
    def forward(self, inputs, **kwargs):
        embed_x = self.embed(inputs, self.L) # B x [3 + L x 2 x 3]
        alpha = (kwargs['global_step'] - kwargs['start']) / (kwargs['end'] - kwargs['start']) * self.L 
        alpha = max(min(alpha, self.L), 0)
        k = torch.arange(self.L,dtype=torch.float32,device=inputs.device)
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        in_channel = embed_x.shape[1] // (1 + 2 * self.L)
        embed_x[:,in_channel:] *= weight[:,None].repeat(1, in_channel * 2).reshape(1,-1)
        return embed_x

class Integrated_Dir_Encoding(nn.Module):
    def __init__(self, deg):
        super(Integrated_Dir_Encoding, self).__init__()
        assert deg <= 5
        ml_array = self.get_ml_array(deg)
        l_max = 2**(deg - 1)

        mat = np.zeros((l_max + 1, ml_array.shape[1]))
        for i, (m, l) in enumerate(ml_array.T):
            for k in range(l - m + 1):
                mat[k, i] = self.sph_harm_coeff(l, m, k)
        
        self.register_buffer("mat", torch.from_numpy(mat).float())
        self.register_buffer("ml_array", torch.from_numpy(ml_array).float())

    def sph_harm_coeff(self, l, m, k):
        def generalized_binomial_coeff(a, k):
            """Compute generalized binomial coefficients."""
            return np.prod(a - np.arange(k)) / np.math.factorial(k)

        def assoc_legendre_coeff(l, m, k):
            """Compute associated Legendre polynomial coefficients.

            Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
            (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

            Args:
                l: associated Legendre polynomial degree.
                m: associated Legendre polynomial order.
                k: power of cos(theta).

            Returns:
                A float, the coefficient of the term corresponding to the inputs.
            """
            return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
                    np.math.factorial(l - k - m) *
                    generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))

        """Compute spherical harmonic coefficients."""
        return (np.sqrt(
            (2.0 * l + 1.0) * np.math.factorial(l - m) /
            (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))


    def get_ml_array(self, deg):
        """Create a list with all pairs of (l, m) values to use in the encoding."""
        ml_list = []
        for i in range(deg):
            l = 2**i
            # Only use nonnegative m values, later splitting real and imaginary parts.
            for m in range(l + 1):
                ml_list.append((m, l))

        # Convert list into a numpy array.
        ml_array = np.array(ml_list).T
        return ml_array
    
    def forward(self, xyz, kappa_inv=None):
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]
        # Compute z Vandermonde matrix.
        vmz = torch.cat([z**i for i in range(self.mat.shape[0])], dim=-1)
        # Compute x+iy Vandermonde matrix.
        vmxy = torch.cat([(x + 1j * y)**m for m in self.ml_array[0, :]], dim=-1)
        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, self.mat)
        # return sph_harms
        # sigma = 0.5 * self.ml_array[1, :] * (self.ml_array[1, :] + 1)
        # ide = sph_harms * torch.exp(-sigma * kappa_inv)
        return torch.cat([torch.real(sph_harms), torch.imag(sph_harms)], dim=-1)


class GeneralMLP(nn.Module):
    def __init__(self, num_in, num_out, activation,
                 hiden_depth=4, hiden_width=64, output_act=False):
        super(GeneralMLP, self).__init__()
        
        assert(hiden_depth >= 1)

        if hiden_depth == 1:
            layers = [nn.Linear(num_in, num_out)]
        else:
            layers = [nn.Linear(num_in, hiden_width)]
            layers.append(activation)
            for i in range(hiden_depth-2):
                layers.append(nn.Linear(hiden_width, hiden_width))
                layers.append(activation)
            layers.append(nn.Linear(hiden_width, num_out))
        if output_act:
            layers.append(activation)
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)






class NeRF(nn.Module):
    def __init__(self, in_channel=3, hiden_width=256, L_loc=10, L_dir=6, activation=nn.ReLU(),
                mode="full"):
        super(NeRF, self).__init__()
        assert mode in ['full','barf',"no"]


        if mode == "full":
            self.PE_loc = Positional_Encoding(L_loc)
            self.PE_dir = Positional_Encoding(L_dir)
        elif mode == 'barf':
            self.PE_loc = Weighted_Positional_Encoding(L_loc)
            self.PE_dir = Weighted_Positional_Encoding(L_dir)
        else:
            L_loc = 0
            L_dir = 0

        self.geo_mlp_1 = GeneralMLP(num_in=L_loc*2*in_channel+in_channel,num_out=256,
                              activation=activation, hiden_depth=5, hiden_width=hiden_width) 
        self.geo_mlp_2 = GeneralMLP(num_in=L_loc*2*in_channel+in_channel+hiden_width,num_out=256,
                              activation=activation, hiden_depth=3, hiden_width=hiden_width) 
        self.color_mlp = GeneralMLP(num_in=L_dir*2*3+3+hiden_width,num_out=3,
                              activation=activation, hiden_depth=2, hiden_width=hiden_width)
        self.linear_sigma = nn.Linear(hiden_width, 1)
        # self.PE_loc = Positional_Encoding(L_loc)
        # self.PE_dir = Positional_Encoding(L_dir)

        self.mode = mode 
    

    def forward(self, **kwargs):
        x = kwargs['x']
        direction = kwargs['direction']
        if self.mode == 'full':
            features1 = self.geo_mlp_1(self.PE_loc(x))
            features2 = self.geo_mlp_2(torch.cat([self.PE_loc(x), features1], -1))
            raw_sigma = self.linear_sigma(features2)
            raw_rgb = self.color_mlp(torch.cat([self.PE_dir(direction), features2], -1))
        elif self.mode == 'barf':
            features1 = self.geo_mlp_1(self.PE_loc(x, **kwargs))
            features2 = self.geo_mlp_2(torch.cat([self.PE_loc(x, **kwargs), features1], -1))
            raw_sigma = self.linear_sigma(features2)
            raw_rgb = self.color_mlp(torch.cat([self.PE_dir(direction, **kwargs), features2], -1))
        else:
            features1 = self.geo_mlp_1(x)
            features2 = self.geo_mlp_2(torch.cat([x, features1], -1))
            raw_sigma = self.linear_sigma(features2)
            raw_rgb = self.color_mlp(torch.cat([direction, features2], -1))    

        return torch.cat([raw_rgb, raw_sigma ], -1)



class PaddingAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask):
        # print(inputs.shape, mask.shape)
        
        temp = torch.where(mask.sum(dim=-1) < mask.shape[1])[0][0:1]
        # print(inputs[temp])
        # print(mask[temp])
        
        padding_inputs_forward(inputs, mask)
        # print("after")
        # print(inputs[temp])
        # input()
        ctx.mask = mask

        return inputs
    
    @staticmethod
    def backward(ctx, grad_in):
        mask = ctx.mask
        padding_inputs_backward(grad_in, mask)
        return grad_in, None

def PaddingInputs(inputs, mask):
    return PaddingAutoGrad.apply(inputs, mask)


class SpecularNet(nn.Module):
    def __init__(self):
        super(SpecularNet, self).__init__()
        self.mlp = GeneralMLP(num_in = 6, num_out = 3, 
                                activation=Gaussian_Act(0.1),hiden_depth=4, hiden_width=64)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        return self.sigmoid(self.mlp(inputs))


# class ColorNet(nn.Module):
#     def __init__(self, bbox_corner, bbox_size):
#         super(ColorNet, self).__init__()
#         self.embedding = PyHashGrid(bbox_corner, bbox_size)
        
        
#         self.mlp = GeneralMLP(num_in = 3, num_out = 1, 
#                               activation=Gaussian_Act(0.1), hiden_depth=4, hiden_width=64)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, inputs):
#         return self.sigmoid(self.mlp(inputs))

        
class ColorNet(nn.Module):
    def __init__(self, num_blend, deg):
        super(ColorNet, self).__init__()

        self.mlp = GeneralMLP(num_in=3, num_out = 16, activation=Gaussian_Act(0.1), hiden_depth=2, hiden_width=32)
        self.mlp2 = GeneralMLP(num_in=16+3, num_out = 1, activation=Gaussian_Act(0.1), hiden_depth=2, hiden_width=32)
        
        
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn. Softplus()
        self.softmax = nn.Softmax(dim=2)
        self.pi = 3.1415926
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, inputs, nei_dir, visibility):
        """
        inputs:  B x num_sample x 32
        nei_dir  B x num_sample x num_neighbor x 3
        """
        
        B, num_sample, num_neighbor = nei_dir.shape[:3]
        
        
        H = self.mlp(inputs)[...,None,:]
        
        coeffi = self.sigmoid(self.mlp2(torch.cat([H.repeat(1,1,num_neighbor,1), nei_dir], -1)))
        
        return coeffi
        
    
class BlendingNet(nn.Module):
    def __init__(self, hiden_depth=4, hiden_width=64, num_blend=2, activation=nn.ReLU()):
        super(BlendingNet, self).__init__()
        
        """
        输入: 
        warped_color  B x num_sample x num_neighbor x 3
        ray_dir       B x num_sample x num_neighbor x 3  
        """

        self.color_map_v1 = GeneralMLP(num_in = num_blend*3, num_out= 3, 
                              activation=Gaussian_Act(0.1),hiden_depth=4, hiden_width=64)
        

        self.num_blend = num_blend

        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn. Softplus()
    

    def forward(self, inputs):
        """
        B x num_sample x num_neighbor x channel
        """
        B, num_sample, num_neighbor = inputs.shape[:3]
        

        return self.sigmoid(self.color_map_v1(inputs.reshape(B, num_sample, -1)))[..., None]


class ProposalNetwork(nn.Module):
    def __init__(self):
        super(ProposalNetwork, self).__init__()
        self.mlp = GeneralMLP(num_in=3, num_out= 1, 
                             activation=Gaussian_Act(0.1),hiden_depth=5, hiden_width=64)
        self.softplus = nn.Softplus()
    def forward(self, x):
        raw = self.mlp(x)
        alpha = self.softplus(raw)
        return alpha
        

class ProposalGaussian(nn.Module):
    def __init__(self, num_view, num_gaussian, height, width):
        super(ProposalGaussian, self).__init__()
        
        self.num_gaussian = num_gaussian
        self.num_view = num_view

        # [TODO] high resolution with large hashmap size ? CHECK  
        self.encoding = HashGrid2D(num_hashgrid=self.num_view,
                                    n_levels=8, 
                                    n_features_per_level=2,
                                    log2_hashmap_size=14,
                                    # base_resolution=(width//32, height//32),
                                    base_resolution=(16,16),
                                    finest_resolution=(width, height))

        self.mlp = GeneralMLP(num_in=16, num_out=self.num_gaussian*3, 
                             activation=Gaussian_Act(0.1),hiden_depth=2, hiden_width=32)
        
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hashidxs, color):
        
        H = self.encoding(x, hashidxs)
        raw = self.mlp(H)    
        
        weight = torch.exp(raw[..., :self.num_gaussian])

        mu = self.sigmoid(raw[..., self.num_gaussian:2*self.num_gaussian])

        inv_sigma = torch.exp(raw[..., 2*self.num_gaussian:3*self.num_gaussian])

        return {"mu": mu, "inv_sigma": inv_sigma, "weight": weight, "rgb": None, "specular": None,
                "feature": H}

            
        
        
class RayGaussian(nn.Module):
    def __init__(self, num_view, num_gaussian, height, width, padding):
        super(RayGaussian, self).__init__()
        
        self.num_gaussian = num_gaussian
        self.num_view = num_view

        
        # [TODO] high resolution with large hashmap size ? CHECK  
        self.encoding = HashGrid2D(num_hashgrid=self.num_view,
                                    n_levels=16, 
                                    n_features_per_level=2,
                                    log2_hashmap_size=16,
                                    # base_resolution=(width//32, height//32),
                                    base_resolution=(16,16),
                                    finest_resolution=(width+2*padding, height+2*padding))


        self.mlp = GeneralMLP(num_in=32, num_out=self.num_gaussian*3, 
                             activation=Gaussian_Act(0.1),hiden_depth=2, hiden_width=64)
        
        self.color_bak = GeneralMLP(num_in=32, num_out=3, 
                             activation=Gaussian_Act(0.1),hiden_depth=2, hiden_width=64)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, hashidxs, nei_dir):
        
        H = self.encoding(x, hashidxs)
        
        
        raw = self.mlp(H)    
        weight = torch.exp(raw[..., :self.num_gaussian])

        mu = self.sigmoid(raw[..., self.num_gaussian:2*self.num_gaussian])
        # inv_sigma = self.softplus(raw[..., 2*self.num_gaussian:3*self.num_gaussian])
        
        inv_sigma = torch.exp(raw[..., 2*self.num_gaussian:3*self.num_gaussian])
        
        warped_pred_color = self.sigmoid(self.color_bak(H))

        # rgb = self.sigmoid(raw[..., 3*self.num_gaussian:])
        return {"mu": mu, "inv_sigma": inv_sigma, "weight": weight, "rgb": None, "specular": None, 
                "warped_pred_color": warped_pred_color, "feature": None}
            
            
                             
                        


def init_model(model, mode = 'default'):
    assert mode in ['xavier', 'kaiming', 'zeros', 'default', "small"]
    def kaiming_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            layer.bias.data.fill_(0.)
    def xavier_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.)
    def zeros_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            layer.bias.data.fill_(0.)
    def small_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.ones_(layer.weight) * 0.0001
            layer.bias.data.fill_(0.0001)
    if mode == 'default':
        return model 
    elif mode == 'kaiming':
        model.apply(kaiming_init)
        print('\n====== Kaiming Init ======\n')
    elif mode == 'xavier':
        model.apply(xavier_init)
        print('\n====== Xavier Init ======\n')
    elif mode == 'zeros':
        model.apply(zeros_init)
        print('\n====== zeros Init ======\n')
    elif mode == "small":
        model.apply(small_init)
        print('\n====== small Init ======\n')
    return model 