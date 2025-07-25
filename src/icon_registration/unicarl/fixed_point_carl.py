import icon_registration as icon
import icon_registration.carl as carl
import icon_registration.data
import icon_registration.networks as networks
from icon_registration.config import device

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch
import torchvision.utils
import torch.nn as nn

def show(tensor):
    plt.imshow(torchvision.utils.make_grid(tensor[:6], nrow=3)[0].cpu().detach())
    plt.xticks([])
    plt.yticks([])
    
MODEL_DIM=384
#MODEL_DIM=128


class SomeDownsampleNoDilationNet(nn.Module):
    def __init__(self, dimension=2, output_dim=MODEL_DIM):
        super().__init__()
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.avg_pool = F.avg_pool2d
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.avg_pool = F.avg_pool3d
        DIM = output_dim
        self.convs = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([self.BatchNorm(DIM) for _ in range(3)])
        self.convs0 = (self.Conv(1, DIM // 4, 2, padding="same"))

        self.convs.append(self.Conv(DIM // 4, DIM, 2, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same"))

    def forward(self, x):
        
        x = self.convs0(x)
        x = self.avg_pool(x, 2, ceil_mode=True)

        x = self.convs[0](x)
        x = self.avg_pool(x, 2, ceil_mode=True)

        x = torch.relu(x)

        for i in range(3):
            x = self.batchnorms[i](x)
            y = self.convs[i + 1](x)
            y = torch.relu(y)
            y = self.convs[i + 4](y)
            y = torch.relu(y)
            y = self.convs[i + 7](y)

            x = y + x

        return x

class SomeDownsampleNet(nn.Module):
    def __init__(self, dimension=2, output_dim=128):
        super().__init__()
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.avg_pool = F.avg_pool2d
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.avg_pool = F.avg_pool3d
        DIM = output_dim
        self.convs = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([self.BatchNorm(DIM) for _ in range(3)])
        self.convs0 = (self.Conv(1, DIM // 4, 2, padding="same"))

        self.convs.append(self.Conv(DIM // 4, DIM, 2, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same", dilation=2))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same", dilation=4))

    def forward(self, x):
        
        x = self.convs0(x)
        x = self.avg_pool(x, 2, ceil_mode=True)

        x = self.convs[0](x)
        x = self.avg_pool(x, 2, ceil_mode=True)

        x = torch.relu(x)

        for i in range(3):
            x = self.batchnorms[i](x)
            y = self.convs[i + 1](x)
            y = torch.relu(y)
            y = self.convs[i + 4](y)
            y = torch.relu(y)
            y = self.convs[i + 7](y)

            x = y + x

        return x


# here be dragons.
# (probably)
z = torch.linalg.inv(torch.tensor([[1.0, 0], [0, 1]]).to(icon_registration.config.device))

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, dimension=2):
        super().__init__()
        dim = 128
        self.dim = dim

        self.dimension=dimension

        self.key_layer = nn.Linear(dim, 64)
        self.value_layer = nn.Linear(dim, 64)
        self.query_layer = nn.Linear(dim, 64)
        self.output_layer = nn.Linear(64, dim)

        self.mlp1 = nn.Linear(dim, dim)
        self.mlp2 = nn.Linear(dim, dim)


        
    def forward(self, ft_A, ft_B):

        shape_reference = ft_A.shape

        
        if self.dimension == 3:
            ft_A = ft_A.reshape(
                -1,
                1,
                self.dim,
                ft_A.shape[-1]*
                ft_A.shape[-2]*
                ft_A.shape[-3],
            )
            ft_B = ft_B.reshape(
                -1,
                1,
                self.dim,
                ft_B.shape[-1]*
                ft_B.shape[-2]*
                ft_B.shape[-3],
            )
        elif self.dimension == 2:
            ft_A = ft_A.reshape(
                -1,
                1,
                self.dim,
                ft_A.shape[-1]*
                ft_A.shape[-2],
            )
            ft_B = ft_B.reshape(
                -1,
                1,
                self.dim,
                ft_B.shape[-1]*
                ft_B.shape[-2],
            )

        
        key = ft_B
        query = ft_A
        value = ft_B
        
        key = key.permute([0, 1, 3, 2]).contiguous()
        query = query.permute([0, 1, 3, 2]).contiguous()
        value = value.permute([0, 1, 3, 2]).contiguous()

        skip = query

        key = self.key_layer(key)
        value = self.value_layer(value)
        query = self.query_layer(query)

        # shape [Batch, Dummy, Space (sequence), Feature]

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
            )

        output = self.output_layer(output)


        output = output + skip

        skip2 = output

        output = self.mlp2(F.relu(self.mlp1(output)))

        output = skip2 + output


        output = output.permute(0, 1, 3, 2)

        if self.dimension == 3:
            output = output.reshape(
                -1,
                self.dim,
                skip.shape[2],
                skip.shape[3],
                skip.shape[4],
            )
        elif self.dimension == 2:
            output = output.reshape(
                -1,
                self.dim,
                shape_reference[2],
                shape_reference[3],
            )
        return output
        

class AttentionFeaturizer(icon_registration.RegistrationModule):
    def __init__(self, net, dimension=2):
        super().__init__()
        self.net = net
        self.dim = 128
        self.dimension =dimension
        self.padding = 16
        self.downscale_factor=4
        self.registerer = carl.RotationFunctionFromVectorField(
            AttentionRegistration(dimension, self.padding // self.downscale_factor)
        )
        self.cross1 = CrossAttentionBlock(dimension=dimension)
        self.cross2 = CrossAttentionBlock(dimension=dimension)


    def crop(self, x):
        padding = self.padding // self.downscale_factor
        if self.dimension == 3:
            return x[:, :, padding:-padding, padding:-padding, padding:-padding]
        elif self.dimension == 2:
            return x[:, :, padding:-padding, padding:-padding]

    def featurize(self, values, recrop=True):
        padding = self.padding
        if self.dimension == 3:
            x = torch.nn.functional.pad(
                values, [padding, padding, padding, padding, padding, padding]
            )
        elif self.dimension == 2:
            x = torch.nn.functional.pad(values, [padding, padding, padding, padding])
        x = self.net(x)
        x = 4 * x / (0.001 + torch.sqrt(torch.sum(x**2, dim=1, keepdims=True)))
        if recrop:
            x = self.crop(x)
        return x

    def forward(self, A, B):

        ft_A = self.featurize(A, recrop=False)
        ft_B = self.featurize(B)

        return self.registerer(ft_A, ft_B)

        #ft_A_cross = self.cross1(ft_A, ft_B)
        #ft_B_cross = self.cross2(ft_B, ft_A)


        #return self.registerer(ft_A_cross, ft_B_cross)
    

class AttentionRegistration(icon_registration.RegistrationModule):
    def __init__(self, dimension=2, padding=8):
        super().__init__()
        self.dim = MODEL_DIM
        self.dimension = dimension

        self.padding = padding

    def torch_attention(self, ft_A, ft_B):
    

        if self.dimension == 3:
            ft_A = ft_A.reshape(
                -1,
                1,
                self.dim,
                (self.identity_map.shape[-1] + 2 * self.padding)
                * (self.identity_map.shape[-2] + 2 * self.padding)
                * (self.identity_map.shape[-3] + 2 * self.padding),
            )
            ft_B = ft_B.reshape(
                -1,
                1,
                self.dim,
                self.identity_map.shape[-1]
                * self.identity_map.shape[-2]
                * self.identity_map.shape[-3],
            )
        elif self.dimension == 2:
            ft_A = ft_A.reshape(
                -1,
                1,
                self.dim,
                (self.identity_map.shape[-1] + 2 * self.padding)
                * (self.identity_map.shape[-2] + 2 * self.padding),
            )
            ft_B = ft_B.reshape(
                -1,
                1,
                self.dim,
                self.identity_map.shape[-1] * self.identity_map.shape[-2],
            )
        ft_A = ft_A.permute([0, 1, 3, 2]).contiguous()
        ft_B = ft_B.permute([0, 1, 3, 2]).contiguous()
        im = carl.pad_im(self.identity_map, self.padding).to(ft_A.device)
        x = im.reshape(-1, 1, self.dimension, ft_A.shape[2]).permute(0, 1, 3, 2)
        x = torch.cat([x, x], axis=-1)
        x = torch.cat([x, x], axis=-1)
        x = x[:, :, :, :4]
        x = x.expand(ft_A.shape[0], -1, -1, -1).contiguous()
        # print(ft_A.stride(), ft_B.stride(), x.stride())
        # print(ft_A.shape, ft_B.shape, x.shape)

        #with torch.backends.cuda.sdp_kernel(enable_math=False):
        output = torch.nn.functional.scaled_dot_product_attention(
            ft_B, ft_A, x, scale=1
        )
        output = output[:, :, :, : self.dimension]
        output = output.permute(0, 1, 3, 2)
        if self.dimension == 3:
            output = output.reshape(
                -1,
                3,
                self.identity_map.shape[2],
                self.identity_map.shape[3],
                self.identity_map.shape[4],
            )
        elif self.dimension == 2:
            output = output.reshape(
                -1,
                2,
                self.identity_map.shape[2],
                self.identity_map.shape[3],
            )
        return output

    def forward(self, A, B):
        
        output = self.torch_attention(A, B)
        if output_saved[0]:
            output_saved[0] = output
        output = output - self.identity_map

        return output

output_saved = [None]



def even_pad(img, amt):
    dimensions = len(img.shape) - 2
    padding = (amt,) * 2 * dimensions
    return F.pad(img, padding, "replicate") * 2 - F.pad(img, padding, "reflect")
from icon_registration.losses import _get_gaussian_kernel1d

def even_blur(tensor, sigma):

    kernel_size = 3 * sigma
    kernel_size += 1 - kernel_size%2
    
    kernel1d = _get_gaussian_kernel1d(kernel_size=kernel_size, sigma=sigma).to(
        tensor.device, dtype=tensor.dtype
    )
    out = tensor
    group = tensor.shape[1]

    out = even_pad(out, kernel_size // 2)

    if len(tensor.shape) - 2 == 1:
        out = torch.conv1d(out, kernel1d[None, None, :].expand(group,-1,-1), groups=group)
    elif len(tensor.shape) - 2 == 2:
        out = torch.conv2d(out, kernel1d[None, None, :, None].expand(group,-1,-1,-1), groups=group)
        out = torch.conv2d(out, kernel1d[None, None, None, :].expand(group,-1,-1,-1), groups=group)
    elif len(tensor.shape) - 2 == 3:
        out = torch.conv3d(out, kernel1d[None, None, :, None, None].expand(group,-1,-1,-1,-1), groups=group)
        out = torch.conv3d(out, kernel1d[None, None, None, :, None].expand(group,-1,-1,-1,-1), groups=group)
        out = torch.conv3d(out, kernel1d[None, None, None, None, :].expand(group,-1,-1,-1,-1), groups=group)


    return out

class Blur(icon.network_wrappers.RegistrationModule):
    def __init__(self, net, radius):
        super().__init__()
        self.radius = radius
        self.net = net
    def forward(self, A, B):
        self.identity_map.isIdentity = True
        phi = self.net(A, B)(self.identity_map)
        phi = even_blur(phi, self.radius)
        field = self.as_function(phi - self.identity_map)

        def transform(coords):
            if hasattr(coords, "isIdentity") and coords.shape[1:] == phi.shape[1:]:
                return phi
            coords_reflected = coords - 2 * coords * (coords < 0) - 2 * (coords - 1) * (coords > 1)
            return coords + 2 * field(coords) - field(coords_reflected)
        return transform

class Equivariantize(icon.RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, a):
        if (len(self.identity_map.shape) == 4) :
            i = self.net(a)
            i = i + self.net(a.flip(dims=(2,3))).flip(dims=(2,3))
            return i / 2
        if (len(self.identity_map.shape) == 5) :
            i = self.net(a)
            i = i + self.net(a.flip(dims=(2, 3))).flip(dims=(2, 3))
            i = i + self.net(a.flip(dims=(3, 4))).flip(dims=(3, 4))
            i = i + self.net(a.flip(dims=(2, 4))).flip(dims=(2, 4))
            return i / 4
        raise NotImplementedError()
