import icon_registration as icon
import torch.nn as nn
import icon_registration.losses
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
import numpy as np
import torch
import torch.linalg
from icon_registration.config import device
from icon_registration.mermaidlite import identity_map_multiN


class NoDownsampleNetBroad(nn.Module):
    def __init__(self, dimension=2, output_dim=128):
        super().__init__()
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
        DIM = output_dim
        self.convs = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([self.BatchNorm(DIM) for _ in range(3)])
        self.convs.append(self.Conv(1, DIM, 3, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same", dilation=2))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same", dilation=4))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same", dilation=8))

    def forward(self, x):
        x = self.convs[0](x)
        x = torch.relu(x)

        for i in range(3):
            x = self.batchnorms[i](x)
            y = self.convs[i + 1](x)
            y = torch.relu(y)
            y = self.convs[i + 4](y)
            y = torch.relu(y)
            y = self.convs[i + 7](y)
            y = torch.relu(y)
            y = self.convs[i + 10](y)

            x = y + x

        return x


class NoDownsampleNet(nn.Module):
    def __init__(self, dimension=2, output_dim=128):
        super().__init__()
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
        DIM = output_dim
        self.convs = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([self.BatchNorm(DIM) for _ in range(3)])
        self.convs.append(self.Conv(1, DIM, 3, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same"))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same", dilation=2))
        for i in range(3):
            self.convs.append(self.Conv(DIM, DIM, 3, padding="same", dilation=4))

    def forward(self, x):
        x = self.convs[0](x)
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
z = torch.linalg.inv(torch.tensor([[1.0, 0], [0, 1]]).cuda())


class RandomMatrix(icon.RegistrationModule):
    def forward(self, a, b):
        if len(a.shape) == 4:
            noise = torch.randn(a.shape[0], 2, 2) * 13
            noise = noise - noise.permute([0, 2, 1])
            noise = torch.linalg.matrix_exp(noise)
            noise = torch.cat([noise, torch.zeros(a.shape[0], 2, 1)], axis=2).to(
                a.device
            )
            x = noise
            x = torch.cat(
                [
                    x,
                    torch.Tensor([[[0, 0, 1]]]).to(x.device).expand(x.shape[0], -1, -1),
                ],
                1,
            )
            x = torch.matmul(
                torch.Tensor([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]).to(x.device), x
            )
            x = torch.matmul(
                x,
                torch.Tensor([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]]).to(x.device),
            )
            return x
        elif len(a.shape) == 5:
            noise = torch.randn(a.shape[0], 3, 3) * 13
            noise = noise - noise.permute([0, 2, 1])
            noise = torch.linalg.matrix_exp(noise)
            noise = torch.cat([noise, torch.zeros(a.shape[0], 3, 1)], axis=2).to(
                a.device
            )
            x = noise
            x = torch.cat(
                [
                    x,
                    torch.Tensor([[[0, 0, 0, 1]]])
                    .to(x.device)
                    .expand(x.shape[0], -1, -1),
                ],
                1,
            )
            x = torch.matmul(
                torch.Tensor(
                    [[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 1]]
                ).to(x.device),
                x,
            )
            x = torch.matmul(
                x,
                torch.Tensor(
                    [[1, 0, 0, -0.5], [0, 1, 0, -0.5], [0, 0, 1, -0.5], [0, 0, 0, 1]]
                ).to(x.device),
            )
            return x


class FunctionsFromMatrix(icon.RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B).detach().clone()
        matrix_phi = np.array(matrix_phi.cpu().detach())
        matrix_phi = torch.tensor(matrix_phi).to(image_A.device)

        def transform(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return icon.network_wrappers.multiply_matrix_vectorfield(
                matrix_phi, coordinates_homogeneous
            )[:, :-1]

        inv = torch.linalg.inv(matrix_phi.detach().clone())

        def invtransform(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return icon.network_wrappers.multiply_matrix_vectorfield(
                inv, coordinates_homogeneous
            )[:, :-1]

        return transform, invtransform


class PostStep(icon.RegistrationModule):

    def __init__(self, netPhi, netPsi):
        super().__init__()
        self.netPhi = netPhi
        self.netPsi = netPsi

    def forward(self, image_A, image_B):

        # Tag for shortcutting hack. Must be set at the beginning of
        # forward because it is not preserved by .to(config.device)
        self.identity_map.isIdentity = True

        phi, invphi = self.netPhi(image_A, image_B)
        psi = self.netPsi(
            image_A,
            self.as_function(image_B)(invphi(self.identity_map)),
        )
        return lambda tensor_of_coordinates: psi(phi(tensor_of_coordinates))


class Equivariantize(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, a):
        i = self.net(a)
        i = i + self.net(a.flip(dims=(2, 3))).flip(dims=(2, 3))
        i = i + self.net(a.flip(dims=(3, 4))).flip(dims=(3, 4))
        i = i + self.net(a.flip(dims=(2, 4))).flip(dims=(2, 4))
        return i / 4


class RotationFunctionFromVectorField(icon.RegistrationModule):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, a, b):
        displacements = self.net(a, b)
        field = self.as_function(displacements)

        def transform(coords):
            coords_reflected = (
                coords - 2 * coords * (coords < 0) - 2 * (coords - 1) * (coords > 1)
            )
            if hasattr(coords, "isIdentity") and coords.shape == displacements.shape:
                return coords + displacemnts
            return coords + 2 * field(coords) - field(coords_reflected)

        return transform


def augmentify(network):
    augmenter = FunctionsFromMatrix(RandomMatrix())
    augmenter2 = icon.FunctionFromMatrix(RandomMatrix())

    augmenter = icon.TwoStepRegistration(
        augmenter2, PostStep(augmenter, network.regis_net)
    )

    network.regis_net = augmenter
    network.assign_identity_map(network.input_shape)
    return network


def make_im(input_shape):
    input_shape = np.array(input_shape)
    input_shape[0] = 1
    spacing = 1.0 / (input_shape[2::] - 1)
    _id = identity_map_multiN(input_shape, spacing)
    return _id


def pad_im(im, n):
    new_shape = np.array(im.shape)
    old_shape = np.array(im.shape)
    new_shape[2:] += 2 * n
    new_im = np.array(make_im(new_shape))
    if len(new_shape) == 4:

        def expand(t):
            return t[None, 2:, None, None]

    else:

        def expand(t):
            return t[None, 2:, None, None, None]

    new_im *= expand((new_shape - 1)) / expand((old_shape - 1))
    new_im -= n / expand((old_shape - 1))
    new_im = torch.tensor(new_im)
    return new_im


class AttentionRegistration(icon_registration.RegistrationModule):
    def __init__(self, net, dimension=2):
        super().__init__()
        self.net = net
        self.dim = 128
        self.dimension = dimension

        self.padding = 9

    def crop(self, x):
        padding = self.padding
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
        im = pad_im(self.identity_map, self.padding).to(ft_A.device)
        x = im.reshape(-1, 1, self.dimension, ft_A.shape[2]).permute(0, 1, 3, 2)
        x = torch.cat([x, x], axis=-1)
        x = torch.cat([x, x], axis=-1)
        x = x[:, :, :, :4]
        x = x.expand(ft_A.shape[0], -1, -1, -1).contiguous()
        # print(ft_A.stride(), ft_B.stride(), x.stride())
        # print(ft_A.shape, ft_B.shape, x.shape)

        with torch.backends.cuda.sdp_kernel(enable_math=False):
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
        ft_A = self.featurize(A, recrop=False)
        ft_B = self.featurize(B)
        output = self.torch_attention(ft_A, ft_B)
        output = output - self.identity_map
        return output

def make_network_final_smolatt(input_shape, dimension, diffusion=False):
    unet = NoDownsampleNet(dimension=dimension)
    ar = AttentionRegistration(unet, dimension=dimension)
    inner_net = icon.network_wrappers.DownsampleRegistration(
        icon.network_wrappers.DownsampleRegistration(
            icon.network_wrappers.DownsampleRegistration(
                icon.FunctionFromVectorField(ar), dimension
            ),
            dimension,
        ),
        dimension,
    )
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(
            icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
            dimension=dimension,
        ),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
    )
    ts = icon.network_wrappers.TwoStepRegistration(inner_net, ts)

    if diffusion:
        net = icon.losses.DiffusionRegularizedNet(ts, icon.LNCC(4), 15)
    else:
        net = icon.losses.GradientICONSparse(ts, icon.LNCC(4), 1.5)

    net.assign_identity_map(input_shape)
    net.cuda()
    return net


def make_network_final_rotation(input_shape, dimension, diffusion=False):
    unet = Equivariantize(NoDownsampleNetBroad(dimension=dimension))
    ar = AttentionRegistration(unet, dimension=dimension)
    inner_net = icon.network_wrappers.DownsampleRegistration(
        icon.network_wrappers.DownsampleRegistration(
            RotationFunctionFromVectorField(ar), dimension
        ),
        dimension,
    )
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(
            icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
            dimension=dimension,
        ),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
    )
    ts = icon.network_wrappers.TwoStepRegistration(inner_net, ts)

    if diffusion:
        net = icon.losses.DiffusionRegularizedNet(inner_net, icon.LNCC(4), 10.5)
    else:
        net = icon.losses.GradientICONSparse(ts, icon.LNCC(4), 1.5)

    net.assign_identity_map(input_shape)
    net.cuda()
    return net


def make_network_final(input_shape, dimension, diffusion=False):
    unet = NoDownsampleNet(dimension=dimension)
    ar = AttentionRegistration(unet, dimension=dimension)
    inner_net = icon.network_wrappers.DownsampleRegistration(
        icon.network_wrappers.DownsampleRegistration(
            icon.FunctionFromVectorField(ar), dimension
        ),
        dimension,
    )
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(
            icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
            dimension=dimension,
        ),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
    )
    ts = icon.network_wrappers.TwoStepRegistration(inner_net, ts)

    if diffusion:
        net = icon.losses.DiffusionRegularizedNet(ts, icon.LNCC(4), 1.5)
    else:
        net = icon.losses.GradientICON(ts, icon.LNCC(4), 1.5)

    net.assign_identity_map(input_shape)
    net.cuda()
    return net


def make_network_final_final(input_shape, dimension, diffusion=False):
    unet = NoDownsampleNet(dimension=dimension)
    ar = AttentionRegistration(unet, dimension=dimension)
    ts = icon.FunctionFromVectorField(ar)
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(ts, dimension=dimension),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
    )
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(ts, dimension=dimension),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
    )

    if diffusion:
        net = icon.losses.DiffusionRegularizedNet(ts, icon.LNCC(4), 1.5)
    else:
        net = icon.losses.GradientICON(ts, icon.LNCC(4), 1.5)

    net.assign_identity_map(input_shape)
    net.cuda()
    return net
