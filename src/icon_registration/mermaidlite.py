# This cell contains the code from https://github.com/uncbiag/mermaid
# that defines the functions compute_warped_image_multiNC
# which we use for composing maps and identity_map_multiN which we use
# to get an identity map.
import numpy as np
import torch


def scale_map(map, sz, spacing):
    """
    Scales the map to the [-1,1]^d format
    :param map: map in BxCxXxYxZ format
    :param sz: size of image being interpolated in XxYxZ format
    :param spacing: spacing of image in XxYxZ format
    :return: returns the scaled map
    """

    ndim = len(spacing)

    # This is to compensate to get back to the [-1,1] mapping of the following form
    # id[d]*=2./(sz[d]-1)
    # id[d]-=1.

    scales = []
    shifts = []

    for d in range(ndim):
        if sz[d + 2] > 1:
            scales.append((2.0 / (sz[d + 2] - 1.0) / spacing[d]))
            shifts.append(-1)       
        else:
            scales.append(1)
            shifts.append(0)
    scales = torch.tensor(scales).float().to(map.device)
    shifts = torch.tensor(shifts).float().to(map.device)


    if ndim == 3:
        scales= scales[:, None, None, None]
        shifts = shifts[:, None, None, None]
    if ndim == 2:
        scales= scales[:, None, None]
        shifts = shifts[:, None, None]
    map_scaled = map * scales + shifts

    return map_scaled


class STNFunction_ND_BCXYZ:
    """
    Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
    """

    def __init__(
        self, spacing, zero_boundary=False, using_bilinear=True, using_01_input=True
    ):
        """
        Constructor
        :param ndim: (int) spatial transformation of the transform
        """
        self.spacing = spacing
        self.ndim = len(spacing)
        # zero_boundary = False
        self.zero_boundary = "zeros" if zero_boundary else "border"
        self.mode = "bilinear" if using_bilinear else "nearest"
        self.using_01_input = using_01_input

    def forward_stn(self, input1, input2, ndim):
        if ndim == 1:
            # use 2D interpolation to mimick 1D interpolation
            # now test this for 1D
            phi_rs = input2.reshape(list(input2.size()) + [1])
            input1_rs = input1.reshape(list(input1.size()) + [1])

            phi_rs_size = list(phi_rs.size())
            phi_rs_size[1] = 2

            phi_rs_ordered = torch.zeros(
                phi_rs_size, dtype=phi_rs.dtype, device=phi_rs.device
            )
            # keep dimension 1 at zero
            phi_rs_ordered[:, 1, ...] = phi_rs[:, 0, ...]

            output_rs = torch.nn.functional.grid_sample(
                input1_rs,
                phi_rs_ordered.permute([0, 2, 3, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
            output = output_rs[:, :, :, 0]

        if ndim == 2:
            # todo double check, it seems no transpose is need for 2d, already in height width design
            input2_ordered = torch.flip(input2, dims=[1])

            if input2_ordered.shape[0] == 1 and input1.shape[0] != 1:
                input2_ordered = input2_ordered.expand(input1.shape[0], -1, -1, -1)
            output = torch.nn.functional.grid_sample(
                input1,
                input2_ordered.permute([0, 2, 3, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
        if ndim == 3:
            input2_ordered = torch.flip(input2, dims=[1])
            if input2_ordered.shape[0] == 1 and input1.shape[0] != 1:
                input2_ordered = input2_ordered.expand(input1.shape[0], -1, -1, -1, -1)
            output = torch.nn.functional.grid_sample(
                input1,
                input2_ordered.permute([0, 2, 3, 4, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
        return output

    def __call__(self, input1, input2):
        """
        Perform the actual spatial transform
        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """

        assert len(self.spacing) + 2 == len(input2.size())
        if self.using_01_input:
            output = self.forward_stn(
                input1, scale_map(input2, input1.shape, self.spacing), self.ndim
            )
        else:
            output = self.forward_stn(input1, input2, self.ndim)
        # print(STNVal(output, ini=-1).sum())
        return output


class STN_ND_BCXYZ:
    """
    Spatial transform code for nD spatial transoforms. Uses the BCXYZ image format.
    """

    def __init__(
        self,
        spacing,
        zero_boundary=False,
        use_bilinear=True,
        use_01_input=True,
        use_compile_version=False,
    ):
        self.spacing = spacing
        """spatial dimension"""
        if use_compile_version:
            if use_bilinear:
                self.f = STNFunction_ND_BCXYZ_Compile(self.spacing, zero_boundary)
            else:
                self.f = partial(get_nn_interpolation, spacing=self.spacing)
        else:
            self.f = STNFunction_ND_BCXYZ(
                self.spacing,
                zero_boundary=zero_boundary,
                using_bilinear=use_bilinear,
                using_01_input=use_01_input,
            )

        """spatial transform function"""

    def __call__(self, input1, input2):
        """
        Simply returns the transformed input
        :param input1: image in BCXYZ format
        :param input2: map in BdimXYZ format
        :return: returns the transformed image
        """
        return self.f(input1, input2)


def compute_warped_image_multiNC(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):
    """Warps image.
    :param I0: image to warp, image size BxCxXxYxZ
    :param phi: map for the warping, size BxdimxXxYxZ
    :param spacing: image spacing [dx,dy,dz]
    :return: returns the warped image of size BxCxXxYxZ
    """

    dim = I0.dim() - 2
    if dim == 1:
        return _compute_warped_image_multiNC_1d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    elif dim == 2:
        return _compute_warped_image_multiNC_2d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    elif dim == 3:
        return _compute_warped_image_multiNC_3d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    else:
        raise ValueError("Images can only be warped in dimensions 1 to 3")


def _compute_warped_image_multiNC_1d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped


def _compute_warped_image_multiNC_2d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped


def _compute_warped_image_multiNC_3d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        # return get_warped_label_map(I0,phi,spacing)
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped


def identity_map_multiN(sz, spacing, dtype="float32"):
    """
    Create an identity map
    :param sz: size of an image in BxCxXxYxZ format
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map
    """
    dim = len(sz) - 2
    nrOfI = int(sz[0])

    if dim == 1:
        id = np.zeros([nrOfI, 1, sz[2]], dtype=dtype)
    elif dim == 2:
        id = np.zeros([nrOfI, 2, sz[2], sz[3]], dtype=dtype)
    elif dim == 3:
        id = np.zeros([nrOfI, 3, sz[2], sz[3], sz[4]], dtype=dtype)
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    for n in range(nrOfI):
        id[n, ...] = identity_map(sz[2::], spacing, dtype=dtype)

    return id


def identity_map(sz, spacing, dtype="float32"):
    """
    Returns an identity map.
    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0 : sz[0]]
    elif dim == 2:
        id = np.mgrid[0 : sz[0], 0 : sz[1]]
    elif dim == 3:
        id = np.mgrid[0 : sz[0], 0 : sz[1], 0 : sz[2]]
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    # now get it into range [0,(sz-1)*spacing]^d
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index

    for d in range(dim):
        id[d] *= spacing[d]

        # id[d]*=2./(sz[d]-1)
        # id[d]-=1.

    # and now store it in a dim+1 array
    if dim == 1:
        idnp = np.zeros([1, sz[0]], dtype=dtype)
        idnp[0, :] = id[0]
    elif dim == 2:
        idnp = np.zeros([2, sz[0], sz[1]], dtype=dtype)
        idnp[0, :, :] = id[0]
        idnp[1, :, :] = id[1]
    elif dim == 3:
        idnp = np.zeros([3, sz[0], sz[1], sz[2]], dtype=dtype)
        idnp[0, :, :, :] = id[0]
        idnp[1, :, :, :] = id[1]
        idnp[2, :, :, :] = id[2]
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    return idnp
