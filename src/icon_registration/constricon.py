import os
import random

from  icon_registration import (losses, networks, DownsampleRegistration, RegistrationModule)
import icon_registration.networks as networks
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from icon_registration.config import device


def find_velocity_fields(phi):
    """
    phi is a function representing a transform, but if it's the integral of a velocity field
    it has that velocity field tacked on to it, ie
    def svf_tranform(coords):
        ....
    svf_transform.velocity_field = velocity_field
    so that it can be picked up here.
    if phi is a composite transform, then it closes over its components.
    """
    
    if hasattr(phi, "velocity_field"):
        yield phi.velocity_field
    for cell in phi.__closure__:
        if hasattr(cell.cell_contents, "__closure__"):
            for elem in find_velocity_fields(cell.cell_contents):
                yield elem


class VelocityFieldBendingEnergyNet(losses.BendingEnergyNet):
    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        fields = list(find_velocity_fields(self.phi_AB))

        return sum(
            losses.BendingEnergyNet.compute_bending_energy_loss(self, field)
            for field in fields
        )


class VelocityFieldDiffusion(losses.DiffusionRegularizedNet):
    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        fields = list(find_velocity_fields(self.phi_AB))

        return sum(
            losses.DiffusionRegularizedNet.compute_bending_energy_loss(
                self, field + self.identity_map
            )
            for field in fields
        )


class ICONSquaringVelocityField(RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.n_steps = 7

    def forward(self, image_A, image_B):
        velocity_field = self.net(image_A, image_B) - self.net(image_B, image_A)
        velocityfield_delta_a = velocity_field / 2**self.n_steps
        velocityfield_delta = velocityfield_delta_a

        for _ in range(self.n_steps):
            velocityfield_delta = velocityfield_delta + self.as_function(
                velocityfield_delta
            )(velocityfield_delta + self.identity_map)

        def transform_AB(coordinate_tensor):
            coordinate_tensor = coordinate_tensor + self.as_function(
                velocityfield_delta
            )(coordinate_tensor)
            return coordinate_tensor

        transform_AB.velocity_field = velocity_field

        velocityfield_delta2 = -velocityfield_delta_a

        for _ in range(self.n_steps):
            velocityfield_delta2 = velocityfield_delta2 + self.as_function(
                velocityfield_delta2
            )(velocityfield_delta2 + self.identity_map)

        def transform_BA(coordinate_tensor):
            coordinate_tensor = coordinate_tensor + self.as_function(
                velocityfield_delta2
            )(coordinate_tensor)
            return coordinate_tensor

        transform_BA.velocity_field = -velocity_field

        return transform_AB, transform_BA

class AntiDiagonalize(RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        m = self.net(image_A, image_B)

        dim = m.shape[1] - 1

        dg = m[:, :dim, :dim]

        dg = dg - torch.transpose(dg, 1, 2)

        dg = torch.cat([dg, torch.zeros(dg.shape[0], 1, dim).to(dg.device)], axis=1)

        dg = torch.cat([dg, m[:, :, dim:]], axis=2)

        return dg

from icon_registration.network_wrappers import multiply_matrix_vectorfield

class FunctionFromMLPWeights(RegistrationModule):
  def __init__(self, net):
    super().__init__()
    self.net = net
    self.hidden_size = 64

  def forward(self, image_A, image_B):
    batch_size = image_A.shape[0]
    network_weights = self.net(image_A, image_B)
    pointer = [0]
    def take(num):
      res = network_weights[:, pointer[0]:pointer[0] + num]
      pointer[0] += num
      return res
    weight_A = take(2 * 64).reshape((batch_size, 64, 2))
    bias_A = take(64)[:, :, None, None]

    weight_B = take(64 * 64).reshape((batch_size, 64, 64))
    bias_B = take(64)[:, :, None, None]

    weight_C = take(64 * 2).reshape((batch_size, 2, 64))
    bias_C = take(2)[:, :, None, None]


    def warp(r):
      feature = multiply_matrix_vectorfield(weight_A, r) + bias_A
      #feature = feature * feature
      feature = torch.nn.functional.gelu(feature)
      feature = multiply_matrix_vectorfield(weight_B, feature) + bias_B
      #feature = feature * feature
      feature = torch.nn.functional.gelu(feature)
      output = multiply_matrix_vectorfield(weight_C, feature) + bias_C
      return output

    return warp

class IntegrateMLP(RegistrationModule):
  def __init__(self, net, steps=6
):
    super().__init__()
    self.net = net
    self.steps=steps
  
  def forward(self, image_A, image_B):
    w1 = self.net(image_A, image_B)
    w2 = self.net(image_B, image_A)

    v = lambda r: w1(r) - w2(r)

    
    def warp(r):
      h = 1 / self.steps
      for i in range(self.steps):
        k1 = v(r)
        k2 = v(r + h * k1 / 2)
        k3 = v(r + h * k2 / 2)
        k4 = v(r + h * k3)

        r = r + h * 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
      return r
    v2 = lambda r: w2(r) - w1(r)

    def warp2(r):
      h = 1 / self.steps
      for i in range(self.steps ):
        k1 = v2(r)
        k2 = v2(r + h * k1 / 2)
        k3 = v2(r + h * k2 / 2)
        k4 = v2(r + h * k3)

        r = r + h * 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
      return r
    return warp, warp2





class ConsistentFromMatrix(RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B) - self.net(image_B, image_A)

        matrix_phi_BA = torch.linalg.matrix_exp(-matrix_phi)
        matrix_phi = torch.linalg.matrix_exp(matrix_phi)

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
            return network_wrappers.multiply_matrix_vectorfield(
                matrix_phi, coordinates_homogeneous
            )[:, :-1]

        def transform1(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return network_wrappers.multiply_matrix_vectorfield(
                matrix_phi_BA, coordinates_homogeneous
            )[:, :-1]

        return transform, transform1


class ExponentialMatrix(RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B) - torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ).to(image_A.device)

        matrix_phi_BA = torch.linalg.matrix_exp(-matrix_phi)
        matrix_phi = torch.linalg.matrix_exp(matrix_phi)

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
            return network_wrappers.multiply_matrix_vectorfield(
                matrix_phi, coordinates_homogeneous
            )[:, :-1]

        def transform1(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return network_wrappers.multiply_matrix_vectorfield(
                matrix_phi_BA, coordinates_homogeneous
            )[:, :-1]

        return transform, transform1


class TwoStepInverseConsistent(RegistrationModule):
    def __init__(self, phi, psi):
        super().__init__()
        self.netPhi = phi
        self.netPsi = psi

    def forward(self, image_A, image_B):
        root_phi_AB, root_phi_BA = self.netPhi(image_A, image_B)

        A_tilde = self.as_function(image_A)(root_phi_AB(self.identity_map))
        B_tilde = self.as_function(image_B)(root_phi_BA(self.identity_map))

        psi_AB, psi_BA = self.netPsi(A_tilde, B_tilde)

        return (
            lambda coord: root_phi_AB(psi_AB(root_phi_AB(coord))),
            lambda coord: root_phi_BA(psi_BA(root_phi_BA(coord))),
        )


class FirstTransform(RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        AB, BA = self.net(image_A, image_B)
        return AB



