import icon_registration as icon
from icon_registration.losses import ICONLoss
import torch

import footsteps

import icon_registration as icon
import icon_registration.networks as networks
import os

from icon_registration import config
from icon_registration.mermaidlite import compute_warped_image_multiNC
import icon_registration.network_wrappers as network_wrappers

class GradientICONFast(icon.RegistrationModule):
    def __init__(self, network, similarity, lmbda):
        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        phi_AB_vectorfield = self.identity_map - phi_AB_vectorfield
        if len(self.identity_map.shape) == 3:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, 1:-1]
            )**2)

        elif len(self.identity_map.shape) == 4:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :-1]
            )**2)
        elif len(self.identity_map.shape) == 5:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :, :-1]
            )**2)


        return bending_energy * self.identity_map.shape[2] **2

    def compute_similarity_measure(self, phi_AB_vectorfield, image_A, image_B):

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        inbounds_tag = torch.zeros(tuple(image_A.shape), device=image_A.device)
        if len(self.input_shape) - 2 == 3:
            inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
        elif len(self.input_shape) - 2 == 2:
            inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
        else:
            inbounds_tag[:, :, 1:-1] = 1.0

        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1)
        )(phi_AB_vectorfield)
        
        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        )
        return similarity_loss

    def forward(self, image_A, image_B) -> ICONLoss:

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)

        self.phi_BA = self.regis_net(image_B, image_A)
        comp = self.phi_BA(self.phi_AB_vectorfield)
        
        similarity_loss = 2 * self.compute_similarity_measure(
            self.phi_AB_vectorfield, image_A, image_B
        )

        bending_energy_loss = self.compute_bending_energy_loss(
            comp
        )

        all_loss = self.lmbda * bending_energy_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            bending_energy_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_AB_vectorfield),
        )

    def prepare_for_viz(self, image_A, image_B):
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA = self.regis_net(image_B, image_A)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        self.warped_image_A = self.as_function(image_A)(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(image_B)(self.phi_BA_vectorfield)

class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def forward(self, image_A, image_B):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return icon.losses.ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            icon.losses.flips(self.phi_BA_vectorfield),
        )


def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5), framework='gradICON'):
    dimension = len(input_shape) - 2
    if framework != 'svf':
        wrapper = icon.FunctionFromVectorField
    else:
        wrapper = icon.network_wrappers.SquaringVelocityField

    inner_net =wrapper(networks.tallUNet2(dimension=dimension))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=dimension),
            wrapper(networks.tallUNet2(dimension=dimension))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension)))
    
    if framework == 'gradICON' or framework == 'svf':
        net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda)
    elif framework == 'icon':
        #TODO: Should we make a sparse version of icon so that it behaves the same as gradicon?
        net = icon.InverseConsistentNet(inner_net, loss_fn, lmbda=lmbda)
    elif framework == 'bendingEnergy':
        net = icon.losses.BendingEnergyNet(inner_net, loss_fn, lmbda=lmbda)
    elif framework == 'svf':
        net = icon.losses.DiffusionRegularizedNet(inner_net, loss_fn, lmbda=lmbda)
    net.assign_identity_map(input_shape)
    return net


def train_two_stage(input_shape, batch_function, GPUS, ITERATIONS_PER_STEP, BATCH_SIZE, framework='gradICON', lmbda=1.5):

    net = make_network(input_shape, include_last_step=False, framework=framework, lmbda=lmbda)

    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    icon.train_batchfunction(net_par, optimizer, batch_function, unwrapped_net=net, steps=ITERATIONS_PER_STEP)
    
    torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + "Step_1_final.trch",
            )

    net_2 = make_network(input_shape, include_last_step=True, framework=framework)

    net_2.regis_net.netPhi.load_state_dict(net.regis_net.state_dict())

    del net
    del net_par
    del optimizer

    if GPUS == 1:
        net_2_par = net_2.cuda()
    else:
        net_2_par = torch.nn.DataParallel(net_2).cuda()
    optimizer = torch.optim.Adam(net_2_par.parameters(), lr=0.00005)

    net_2_par.train()
    
    # We're being weird by training two networks in one script. This hack keeps
    # the second training from overwriting the outputs of the first.
    footsteps.output_dir_impl = footsteps.output_dir + "2nd_step/"
    os.makedirs(footsteps.output_dir)

    icon.train_batchfunction(net_2_par, optimizer, batch_function, unwrapped_net=net_2, steps=ITERATIONS_PER_STEP )
    
    torch.save(
                net_2.regis_net.state_dict(),
                footsteps.output_dir + "Step_2_final.trch",
            )
