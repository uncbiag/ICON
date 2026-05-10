from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


class Warp(nn.Module):
    """Warp an image with given flow / dense displacement field (DDF).

    Args:
        image_size (Sequence[int]): size of input image.
        interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]
    """

    def __init__(self,
                 image_size: Sequence[int],
                 interp_mode: str = 'bilinear') -> None:
        super().__init__()

        self.ndim = len(image_size)
        self.image_size = image_size
        self.interp_mode = interp_mode

        # create reference grid
        grid = self.get_reference_grid(image_size)
        grid = grid.unsqueeze(0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid, persistent=False)

    @staticmethod
    def get_reference_grid(image_size: Sequence[int]) -> torch.Tensor:
        """
        Generate unnormalized reference coordinate grid.
        Args:
            image_size (Sequence[int]): size of input image

        Returns:
            grid: torch.FloatTensor

        """
        mesh_points = [
            torch.arange(0, dim, dtype=torch.float) for dim in image_size
        ]
        grid = torch.stack(torch.meshgrid(*mesh_points),
                           dim=0)  # (spatial_dims, ...)
        return grid

    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp image with flow.
        Args:
            image (torch.Tensor): input image of shape [batch_size, channels, ...]
            flow (torch.Tensor): flow field of shape [batch_size, spatial_dims, ...]

        Returns:
            torch.Tensor: Warped image.
        """
        assert list(self.image_size) == list(image.shape[2:]) == list(flow.shape[2:])

        # deformation
        # [BNHWD]
        sample_grid = self.grid + flow

        # normalize
        # F.grid_sample takes normalized grid with range of [-1,1]
        for i, dim in enumerate(self.image_size):
            sample_grid[:, i, ...] = sample_grid[:, i, ...] * 2 / (dim - 1) - 1

        # [BNHWD] -> [BHWDN]
        # [X,Y,Z, [x,y,z]]
        sample_grid = sample_grid.permute([0] + list(range(2, 2 + self.ndim)) +
                                          [1])
        index_ordering: List[int] = list(range(self.ndim - 1, -1, -1))
        # F.grid_sample takes grid in a reverse order
        sample_grid = sample_grid[..., index_ordering]  # x,y,z -> z,y,x

        return F.grid_sample(image,
                             sample_grid,
                             align_corners=True,
                             mode=self.interp_mode)

    def __repr__(self) -> str:
        return (self.__class__.__name__ + f'(image_size={self.image_size}, '
                                          f'interp_mode={self.interp_mode})')


class Warp_off_grid(Warp):
    """Off-grid version of Warp module.

    Args:
        image_size (Sequence[int]): size of input image.
        interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]
    """

    def __init__(self,
                 image_size: Sequence[int],
                 interp_mode: str = 'bilinear') -> None:
        super().__init__(image_size, interp_mode)

    def forward(self,
                image: torch.Tensor,
                flow: torch.Tensor,
                epsilon: torch.Tensor) -> torch.Tensor:
        """
        Warp image with flow.
        Args:
            image (torch.Tensor): input image of shape [batch_size, channels, ...]
            flow (torch.Tensor): flow field of shape [batch_size, spatial_dims, ...]
            epsilon (torch.Tensor): off-grid noise

        Returns:
            torch.Tensor: Warped image.
        """
        assert (list(self.image_size) ==
                list(image.shape[2:]) ==
                list(flow.shape[2:]) ==
                list(epsilon.shape[2:]))

        # deformation
        # [BNHWD]
        sample_grid = self.grid + flow + epsilon

        # normalize
        # F.grid_sample takes normalized grid with range of [-1,1]
        for i, dim in enumerate(self.image_size):
            sample_grid[:, i, ...] = sample_grid[:, i, ...] * 2 / (dim - 1) - 1

        # [BNHWD] -> [BHWDN]
        # [X,Y,Z, [x,y,z]]
        sample_grid = sample_grid.permute([0] + list(range(2, 2 + self.ndim)) +
                                          [1])
        index_ordering: List[int] = list(range(self.ndim - 1, -1, -1))
        # F.grid_sample takes grid in a reverse order
        sample_grid = sample_grid[..., index_ordering]  # x,y,z -> z,y,x

        return F.grid_sample(image,
                             sample_grid,
                             align_corners=True,
                             mode=self.interp_mode)


@LOSSES.register_module()
@LOSSES.register_module('IntensityLoss')
class FlowLoss(nn.Module):
    """Compute the flow loss between the predicted flow and the ground truth flow.

    Args:
        penalty (str): The penalty norm to use. Options: ['l1', 'l2', 'rmse', 'charbonnier'].
        ch_cfg (CFG): The config for the Charbonnier penalty.
    """
    def __init__(self, penalty: str = 'l2', ch_cfg: Optional[CFG] = None):
        super().__init__()
        self.penalty = penalty
        self.ch_cfg = {} if ch_cfg is None else ch_cfg

    def forward(self,
                pred_flow: torch.Tensor,
                gt_flow: torch.Tensor,
                fg_mask: Optional[torch.Tensor] = None,
                val: bool=False) -> torch.Tensor:
        """
        Args:
            pred_flow (torch.Tensor): The predicted flow. Tensor of shape [B3HWD].
            gt_flow (torch.Tensor): The ground truth flow. Tensor of shape [B3HWD].
            fg_mask (torch.Tensor): The foreground mask in target space. Tensor of shape [B1HWD].
            val (bool): If True, keep the batch dimension of the computed loss.
        """
        if self.penalty == 'l1':
            dist = torch.sum(torch.abs(pred_flow - gt_flow),
                             dim=1,
                             keepdim=True)
        elif self.penalty == 'l2':
            dist = torch.sum((pred_flow - gt_flow)**2, dim=1, keepdim=True)
        elif self.penalty == 'rmse':
            dist = torch.linalg.norm((pred_flow - gt_flow),
                                     dim=1,
                                     keepdim=True)
        elif self.penalty == 'charbonnier':
            dist = charbonnier_loss(pred_flow, gt_flow, **self.ch_cfg)
        else:
            raise ValueError(
                f'Unsupported norm: {self.penalty}, available options are ["l1","l2", "rmse", "charbonnier"].'
            )

        # dist: (B1HWD)
        # fg_mask: (B1HWD)
        if fg_mask is not None:
            if dist.shape[-3:] != fg_mask.shape[:-3]:
                output_size = dist.shape[-3:]
                fg_mask = F.interpolate(fg_mask,
                                        align_corners=True,
                                        size=output_size,
                                        mode='trilinear')

            if dist.shape[0] != fg_mask.shape[0]:
                fg_mask = fg_mask.repeat(dist.shape[0], 1, 1, 1, 1)

            assert dist.shape == fg_mask.shape

            if not val:
                loss = torch.sum(dist * fg_mask) / torch.sum(fg_mask)
            else:
                loss = (dist*fg_mask).sum(dim=(2,3,4)) / fg_mask.sum(dim=(2,3,4))
        else:
            if not val:
                loss = torch.mean(dist)
            else:
                loss = dist.mean(dim=(2,3,4))

        return loss

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(penalty=\'{self.penalty}\',' f'ch_cfg={self.ch_cfg})')
        return repr_str


@LOSSES.register_module()
class InverseConsistentLoss(nn.Module):
    def __init__(
        self,
        flow_loss_cfg: CFG,
        image_size: Sequence[int] = (160, 192, 224),
        interp_mode: str = 'bilinear',
    ):
        """
        Compute the inverse consistency loss of forward and backward flow
        Args:
            image_size (Sequence[int]): shape of input flow field.
        """
        super().__init__()
        flow_loss_cfg.pop('type', None)
        self.flow_loss = FlowLoss(**flow_loss_cfg)
        self.image_size = image_size
        self.interp_mode = interp_mode
        self.warp = Warp(self.image_size, self.interp_mode)

    def forward(
        self,
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
        target_fg: Optional[torch.Tensor] = None,
        source_fg: Optional[torch.Tensor] = None,
        val: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            forward_flow: in TARGET space, mapping from TARGET space to SOURCE space. Tensor of shape [B3HWD].
            backward_flow: in SOURCE spacce, mapping from SOURCE space to TARGET space. Tensor of shape [B3HWD].
        """

        # backward_flow in TARGET space
        backward_flow_ = self.warp(backward_flow, forward_flow)
        # forward_flow in SOURCE space
        forward_flow_ = self.warp(forward_flow, backward_flow)

        zero_flow = torch.zeros_like(forward_flow)

        # forward_flow + backward_flow_ = 0
        # backward_flow + forward_flow_ = 0
        loss = (self.flow_loss(forward_flow + backward_flow_, zero_flow,
                               target_fg, val) +
                self.flow_loss(backward_flow + forward_flow_, zero_flow,
                               source_fg, val))

        return loss, forward_flow_, backward_flow_

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss}, '
                     f'image_size={self.image_size}, '
                     f'interp_mode={self.interp_mode})')
        return repr_str


@LOSSES.register_module()
class ICONLoss(InverseConsistentLoss):
    def __init__(self,
                 flow_loss_cfg: CFG,
                 image_size: Sequence[int] = (160, 192, 224),
                 interp_mode: str = 'bilinear',
                 ):
        """
        Compute the inverse consistency loss of forward and backward flow
        Args:
            image_size (Sequence[int]): shape of input flow field.
        """
        super().__init__(flow_loss_cfg, image_size, interp_mode)
        self.warp_off = Warp_off_grid(self.image_size, self.interp_mode)

    def forward(
            self,
            forward_flow: torch.Tensor,
            backward_flow: torch.Tensor,
            target_fg: Optional[torch.Tensor] = None,
            source_fg: Optional[torch.Tensor] = None,
            val: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            forward_flow: in TARGET space, mapping from TARGET space to SOURCE space. Tensor of shape [B3HWD].
            backward_flow: in SOURCE spacce, mapping from SOURCE space to TARGET space. Tensor of shape [B3HWD].
        """

        # Gaussian noise for off-grid sampling
        epsilon = torch.randn_like(forward_flow) * (1.0 / self.image_size[-1])

        # off_grid forward_flow
        # phi_AB(I + epsilon)
        fwd_flow_eps = self.warp(forward_flow, epsilon)
        # off_grid backward_flow
        # phi_BA(I + epsilon)
        bck_flow_eps = self.warp(backward_flow, epsilon)

        # off_grid backward_flow in TARGET space
        # phi_BA(phi_AB(I + epsilon)+I+epsilon)
        bck_flow_eps_ = self.warp_off(backward_flow, fwd_flow_eps, epsilon)
        # off_grid forward_flow in SOURCE space
        # phi_AB(phi_BA(I + epsilon)+I+epsilon)
        fwd_flow_eps_ = self.warp_off(forward_flow, bck_flow_eps, epsilon)

        zero_flow = torch.zeros_like(forward_flow)

        # fwd_flow_eps + bck_flow_eps_ = 0
        # phi_AB(I + epsilon) + phi_BA(phi_AB(I + epsilon)+I+epsilon) + (I+epsilon) - (I+epsilon) = 0
        # bck_flow_eps + fwd_flow_eps_ = 0
        # phi_BA(I + epsilon) + phi_AB(phi_BA(I + epsilon)+I+epsilon) + (I+epsilon) - (I+epsilon) = 0
        loss = (self.flow_loss(fwd_flow_eps + bck_flow_eps_, zero_flow,
                               target_fg, val) +
                self.flow_loss(bck_flow_eps + fwd_flow_eps_, zero_flow,
                               source_fg, val))

        return loss#, fwd_flow_eps, bck_flow_eps, fwd_flow_eps_, bck_flow_eps_

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss}, '
                     f'image_size={self.image_size}, '
                     f'interp_mode={self.interp_mode})')
        return repr_str


@LOSSES.register_module()
class GradICONLoss(ICONLoss):
    def __init__(self,
                 flow_loss_cfg: CFG,
                 image_size: Sequence[int] = (160, 192, 224),
                 interp_mode: str = 'bilinear',
                 delta: float = 0.001,
                 ):
        """
        Compute the inverse consistency loss of forward and backward flow
        Args:
            image_size (Sequence[int]): shape of input flow field.
        """
        super().__init__(flow_loss_cfg, image_size, interp_mode)
        self.ndim = len(image_size)
        self.delta = delta

    def forward(
            self,
            forward_flow: torch.Tensor,
            backward_flow: torch.Tensor,
            target_fg: Optional[torch.Tensor] = None,
            source_fg: Optional[torch.Tensor] = None,
            val: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            forward_flow: in TARGET space, mapping from TARGET space to SOURCE space. Tensor of shape [B3HWD].
            backward_flow: in SOURCE spacce, mapping from SOURCE space to TARGET space. Tensor of shape [B3HWD].
        """

        # Gaussian noise for off-grid sampling
        epsilon = torch.randn_like(forward_flow) * (1.0 / self.image_size[-1])

        # off_grid forward_flow
        # phi_AB(I + epsilon)
        fwd_flow_eps = self.warp(forward_flow, epsilon)
        # off_grid backward_flow
        # phi_BA(I + epsilon)
        bck_flow_eps = self.warp(backward_flow, epsilon)

        # off_grid backward_flow in TARGET space
        # phi_BA(phi_AB(I + epsilon)+I+epsilon)
        bck_flow_eps_ = self.warp_off(backward_flow, fwd_flow_eps, epsilon)
        # off_grid forward_flow in SOURCE space
        # phi_AB(phi_BA(I + epsilon)+I+epsilon)
        fwd_flow_eps_ = self.warp_off(forward_flow, bck_flow_eps, epsilon)

        # inverse consistency error in TARGET space
        # fwd_flow_eps + bck_flow_eps_ = 0
        # phi_AB(I + epsilon) + phi_BA(phi_AB(I + epsilon)+I+epsilon) = 0
        tgt_ic_err = fwd_flow_eps + bck_flow_eps_
        # inverse consistency error in SOURCE space
        src_ic_err = bck_flow_eps + fwd_flow_eps_

        loss = 0.0

        for i in range(self.ndim):
            d = torch.zeros([1] + [self.ndim] + [1] * self.ndim)
            d[:, i, ...] = self.delta

            fwd_flow_eps_d = self.warp(forward_flow, epsilon + d)
            bck_flow_eps_d = self.warp(backward_flow, epsilon + d)

            bck_flow_eps_d_ = self.warp_off(backward_flow, fwd_flow_eps_d, epsilon + d)
            fwd_flow_eps_d_ = self.warp_off(forward_flow, bck_flow_eps_d, epsilon + d)

            # inverse consistency error (with delta) in TARGET space
            tgt_ic_err_d = fwd_flow_eps_d + bck_flow_eps_d_
            # inverse consistency error (with delta) in SOURCE space
            src_ic_err_d = bck_flow_eps_d + fwd_flow_eps_d_

            tgt_gradicon_err = (tgt_ic_err - tgt_ic_err_d) / self.delta
            src_gradicon_err = (src_ic_err - src_ic_err_d) / self.delta

            loss += torch.mean(tgt_gradicon_err ** 2) + torch.mean(src_gradicon_err ** 2)

        return loss

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flow_loss={self.flow_loss}, '
                     f'image_size={self.image_size}, '
                     f'interp_mode={self.interp_mode}, '
                     f'delta={self.delta})')
