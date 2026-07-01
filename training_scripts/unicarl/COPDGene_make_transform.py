import argparse
import os
import random
import unittest

import footsteps
import icon_registration as icon
import icon_registration.data
import icon_registration.itk_wrapper
import icon_registration.networks as networks
import icon_registration.pretrained_models
import icon_registration.pretrained_models.lung_ct
import icon_registration.test_utils
import itk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
from icon_registration.config import device

footsteps.initialize(output_root="evaluation_results/")
import utils

input_shape = [1, 1, 175, 175, 175]


image_root = "/playpen-raid1/lin.tian/data/lung/dirlab_highres_350"
landmark_root = "/playpen-raid1/lin.tian/data/lung/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/landmarks/"

cases = [f"copd{i}_highres" for i in range(1, 11)]


parser = argparse.ArgumentParser()
parser.add_argument("weights_path")
parser.add_argument("--finetune", action="store_true")
args = parser.parse_args()
weights_path = args.weights_path

#net = equivariant_reg.make_network_final(input_shape, 3)


#threestep_consistent_net = equivariant_reg.make_network_final(input_shape, dimension=3, diffusion=False)
#threestep_consistent_net.regis_net = icon.TwoStepRegistration(threestep_consistent_net.regis_net,
#        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=3)))

#threestep_consistent_net.assign_identity_map(input_shape)
#net = threestep_consistent_net
#p

import register
net = register.get_model()

import icon_registration.network_wrappers


#utils.log(net.regis_net.load_state_dict(torch.load(weights_path)))

#net.regis_net = net.regis_net.phi

net.eval()

overall_1 = []
overall_2 = []
flips = []
ICON_errors=[]

for case in cases[:]:
    image_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_img.nii.gz")
    image_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_img.nii.gz")
    seg_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_label.nii.gz")
    seg_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_label.nii.gz")

    landmarks_insp = icon_registration.test_utils.read_copd_pointset(
        landmark_root + f"/{case.split('_')[0]}_300_iBH_xyz_r1.txt"
    )
    landmarks_exp = icon_registration.test_utils.read_copd_pointset(
        landmark_root + f"/{case.split('_')[0]}_300_eBH_xyz_r1.txt"
    )

    image_insp_preprocessed = register.preprocess(image_insp)
    image_exp_preprocessed = register.preprocess(image_exp)

    image_insp_preprocessed = (
        icon_registration.pretrained_models.lung_network_preprocess(
            image_insp, seg_insp
        )
    )
    image_exp_preprocessed = (
        icon_registration.pretrained_models.lung_network_preprocess(image_exp, seg_exp)
    )

#    phi_AB, phi_BA, loss = icon_registration.itk_wrapper.register_pair(
#        net,
#        image_insp_preprocessed,
#        image_exp_preprocessed,
#        finetune_steps=(50 if args.finetune == True else None),
#        return_artifacts=True,
#    )
    phi_BA, phi_AB, loss = icon_registration.itk_wrapper.register_pair(
        net,
        image_exp_preprocessed,
        image_insp_preprocessed,
        finetune_steps=(50 if args.finetune == True else None),
        return_artifacts=True,
    )
    dists = []
    for i in range(len(landmarks_exp)):
        px, py = (
            landmarks_insp[i],
            np.array(phi_AB.TransformPoint(tuple(landmarks_exp[i]))),
        )
        dists.append(np.sqrt(np.sum((px - py) ** 2)))
    utils.log(f"Mean error on {case}: ", np.mean(dists))
    overall_1.append(np.mean(dists))
    dists = []
    for i in range(len(landmarks_insp)):
        px, py = (
            landmarks_exp[i],
            np.array(phi_BA.TransformPoint(tuple(landmarks_insp[i]))),
        )
        dists.append(np.sqrt(np.sum((px - py) ** 2)))
    utils.log(f"Mean error on {case}: ", np.mean(dists))

    overall_2.append(np.mean(dists))

    utils.log("flips:", loss.flips)

    flips.append(loss.flips)

    scale = 175
    zz = (net.phi_AB(net.phi_BA(net.identity_map)) - net.identity_map) * scale
    icon_error = torch.mean(torch.sqrt(torch.sum(zz**2, axis=1))).item()
    ICON_errors.append(icon_error)
    utils.log("ICON_error", icon_error)

    itk.imwrite(image_insp, footsteps.output_dir + "moving_image.nrrd")
    itk.imwrite(image_insp_preprocessed, footsteps.output_dir + "preprocessed_moving.nrrd")
    itk.imwrite(image_exp, footsteps.output_dir + "fixed.nrrd")
    itk.imwrite(image_exp_preprocessed, footsteps.output_dir + "preprocessed_fixed.nrrd")
    itk.transformwrite([phi_AB], footsteps.output_dir + "transform.hdf5")

utils.log("mean ICON error", np.mean(ICON_errors))


utils.log("overall:")
utils.log(np.mean(overall_1))
utils.log(np.mean(overall_2))
utils.log("flips:", np.mean(flips))
utils.log("flips / prod(imnput_shape", np.mean(flips) / np.prod(input_shape))
