import os
import random
from datetime import datetime

import footsteps
import numpy as np
import torch
import torch.nn.functional as F
from dataset import COPDDataset, HCPDataset, OAIDataset
from torch.utils.data import ConcatDataset, DataLoader

import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration import config
from icon_registration.losses import ICONLoss, to_floats
from icon_registration.mermaidlite import compute_warped_image_multiNC


def write_stats(writer, stats: ICONLoss, ite):
    for k, v in to_floats(stats)._asdict().items():
        writer.add_scalar(k, v, ite)

input_shape = [1, 1, 175, 175, 175]

BATCH_SIZE=4
GPUS = 4

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


def get_dataset():
    return ConcatDataset(
        (
        OAIDataset(input_shape[2:]),
        HCPDataset(input_shape[2:]),
        COPDDataset(
            "/playpen-raid2/lin.tian/projects/icon_lung/ICON_lung/splits/train.txt",
            desired_shape=input_shape[2:])
        )
    )

def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5)):
    dimension = len(input_shape) - 2
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=dimension),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension)))

    net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda)
    net.assign_identity_map(input_shape)
    return net

def train_kernel(optimizer, net, moving_image, fixed_image, writer, ite):
    optimizer.zero_grad()
    loss_object = net(moving_image, fixed_image)
    loss = torch.mean(loss_object.all_loss)
    loss.backward()
    optimizer.step()
    print(to_floats(loss_object))
    write_stats(writer, loss_object, ite)

def train(
    net,
    optimizer,
    data_loader,
    epochs=200,
    eval_period=-1,
    save_period=-1,
    step_callback=(lambda net: None),
    unwrapped_net=None,
):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """
    import footsteps
    from torch.utils.tensorboard import SummaryWriter

    if unwrapped_net is None:
        unwrapped_net = net

    loss_curve = []
    writer = SummaryWriter(
        footsteps.output_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )

    iteration = 0
    for epoch in range(epochs):
        for moving_image, fixed_image in data_loader:
            moving_image, fixed_image = moving_image.cuda(), fixed_image.cuda()
            train_kernel(optimizer, net, moving_image, fixed_image,
                         writer, iteration)
            iteration += 1

            step_callback(unwrapped_net)
        
        
        if epoch % save_period == 0:
            torch.save(
                optimizer.state_dict(),
                footsteps.output_dir + "checkpoints/optimizer_weights_" + str(epoch),
            )
            torch.save(
                unwrapped_net.regis_net.state_dict(),
                footsteps.output_dir + "checkpoints/network_weights_" + str(epoch),
            )

        if epoch % eval_period == 0:
            visualization_moving, visualization_fixed = next(iter(data_loader))
            visualization_moving, visualization_fixed = visualization_moving.cuda(), visualization_fixed.cuda()
            unwrapped_net.eval()
            print("val (from train set)")
            warped = []
            with torch.no_grad():
                print( unwrapped_net(visualization_moving, visualization_fixed))
                warped = unwrapped_net.warped_image_A.cpu()
            unwrapped_net.train()

            def render(im):
                if len(im.shape) == 5:
                    im = im[:, :, :, im.shape[3] // 2]
                if torch.min(im) < 0:
                    im = im - torch.min(im)
                if torch.max(im) > 1:
                    im = im / torch.max(im)
                return im[:4, [0, 0, 0]].detach().cpu()

            writer.add_images(
                "moving_image", render(visualization_moving[:4]), epoch, dataformats="NCHW"
            )
            writer.add_images(
                "fixed_image", render(visualization_fixed[:4]), epoch, dataformats="NCHW"
            )
            writer.add_images(
                "warped_moving_image",
                render(warped),
                epoch,
                dataformats="NCHW",
            )
            writer.add_images(
                "difference",
                render(torch.clip((warped[:4, :1] - visualization_fixed[:4, :1].cpu()) + 0.5, 0, 1)),
                epoch,
                dataformats="NCHW",
            )

def train_two_stage(input_shape, data_loader, GPUS, epochs, eval_period, save_period):

    net = make_network(input_shape, include_last_step=False)

    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    train(net_par, optimizer, data_loader, unwrapped_net=net, 
          epochs=epochs, eval_period=eval_period, save_period=save_period)
    
    torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + "checkpoints/Step_1_final.trch",
            )

    # net_2 = make_network(input_shape, include_last_step=True, framework=framework)

    # net_2.regis_net.netPhi.load_state_dict(net.regis_net.state_dict())

    # del net
    # del net_par
    # del optimizer

    # if GPUS == 1:
    #     net_2_par = net_2.cuda()
    # else:
    #     net_2_par = torch.nn.DataParallel(net_2).cuda()
    # optimizer = torch.optim.Adam(net_2_par.parameters(), lr=0.00005)

    # net_2_par.train()
    
    # # We're being weird by training two networks in one script. This hack keeps
    # # the second training from overwriting the outputs of the first.
    # footsteps.output_dir_impl = footsteps.output_dir + "2nd_step/"
    # os.makedirs(footsteps.output_dir)

    # train(net_2_par, optimizer, data_loader, unwrapped_net=net_2, epochs=100, save_period=10)
    
    # torch.save(
    #             net_2.regis_net.state_dict(),
    #             footsteps.output_dir + "Step_2_final.trch",
    #         )

if __name__ == "__main__":
    footsteps.initialize()

    dataloader = DataLoader(
        get_dataset(),
        batch_size=BATCH_SIZE*GPUS,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    os.makedirs(footsteps.output_dir + "checkpoints", exist_ok=True)

    train_two_stage(input_shape, dataloader, GPUS, 200, 1, 10)