import argparse
import icon_registration.itk_wrapper
import csv
import os

import footsteps
import icon_registration as icon
import itk
import numpy as np
import torch
import torch.nn.functional as F
import utils
from scipy.ndimage.interpolation import zoom as zoom
from tqdm import tqdm
import register


import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, help="the path to the folder containing learn2reg AbdomenMRCT dataset")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")


args = parser.parse_args()
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)


#if args.exp == "":
#    footsteps.initialize(output_root="evaluation_results/")
#else:
#    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/L2R_abdomen")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

os.makedirs(f"{footsteps.output_dir}/submission/task_01", exist_ok=True)

net = register.get_model()

net.to(device)
net.eval()

import json
import os

import itk
import nibabel as nib


def mean_dice(im1, im2):
    array1 = im1
    array2 = im2
    dices = []
    for index in range(1, max(np.max(array1), np.max(array2)) + 1):
        m1 = array1 == index
        m2 = array2 == index
        
        intersection = np.logical_and(m1, m2)
        
        d = 2 * np.sum(intersection) / (np.sum(m1) + np.sum(m2))
        dices.append(d)
    return np.mean(dices)

with open(f"{args.data_folder}/AbdomenMRCT_dataset.json", 'r') as data_info:
     data_info = json.loads(data_info.read())
     test_cases = [[c["fixed"], c["moving"]] for c in data_info["registration_val"]]
     test_names = [12, 14, 16]

     for i in range(3):
         test_cases[i].append(test_names[i])

print(test_cases)


#with open(f"{args.data_folder}/pairs_val.csv", 'r') as data_info:
#    csv_reader = csv.reader(data_info)
#    next(csv_reader)
#    test_cases = [[f"Training/img/img{int(row[0]):04d}.nii.gz", f"Training/img/img{int(row[1]):04d}.nii.gz"] for row in csv_reader]

#spacing = 1.0 / (np.array(origin_shape[2::]) - 1)
#identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(origin_shape, spacing)).to(device)

import affine_decomposition

flips = []
original_state_dict = net.state_dict()
for (fixed_path, moving_path, output_name) in tqdm(test_cases[:]):
    # Restore net weight in case we ran IO
    net.load_state_dict(original_state_dict)
    net = register.get_model()

    net.to(device)
    net.eval()
    

    raw_fixed = itk.imread(os.path.join(args.data_folder, fixed_path))
    raw_moving = itk.imread(os.path.join(args.data_folder, moving_path))

    phi_AB, phi_BA,  artifact = icon.itk_wrapper.register_pair(net, register.preprocess(raw_moving), register.preprocess(raw_fixed), finetune_steps=300, return_artifacts=True)
    flips.append(artifact.flips)
    phi_AB = affine_decomposition.decompose_icon_itk_transform(phi_AB)

    dispfield_filter = itk.TransformToDisplacementFieldFilter[itk.Image[itk.Vector[itk.F, 3], 3], itk.D].New()

    dispfield_filter.SetTransform(phi_AB)
    dispfield_filter.SetReferenceImage(raw_fixed)
    dispfield_filter.SetUseReferenceImage(True)

    dispfield_filter.Update()

    displacement_field = dispfield_filter.GetOutput()

    disp2 = itk.GetArrayFromImage(displacement_field).transpose(3, 2, 1, 0)

    disp_x = -zoom(disp2[0], 0.5, order=2).astype('float16')
    disp_y = -zoom(disp2[1], 0.5, order=2).astype('float16')
    disp_z = zoom(disp2[2], 0.5, order=2).astype('float16')
    disp2 =  np.array((disp_x, disp_y, disp_z)) / 2

    np.savez_compressed(f"{footsteps.output_dir}/submission/task_01/disp_00{output_name}_00{output_name}.npz", disp2)
    #print(disp.shape, disp2.shape)
    #
    #print("Magnitude")
    #print(np.sqrt(np.mean(disp**2)))
    #print("Error")
    #print(np.sqrt(np.mean((disp - disp2)**2)))
    #
    #print("orig", [np.mean(disp[i]) for i in range(3)])
    #print("new", [np.mean(disp2[i]) for i in range(3)])

    #itk.imwrite(raw_fixed, footsteps.output_dir + "raw_fixed.nrrd")
    #itk.imwrite(raw_moving, footsteps.output_dir + "raw_moving.nrrd")
    #itk.imwrite(preprocess(raw_fixed), footsteps.output_dir + "preprocessed_raw_fixed.nrrd")
    #itk.imwrite(preprocess(raw_moving), footsteps.output_dir + "preprocessed_raw_moving.nrrd")
    #
    #itk.transformwrite([phi_AB], footsteps.output_dir + "transform.hdf5")
    #break
    #
    ## Save to output folders
    # disp_itk_format = nib.Nifti1Image(disp_itk_format, affine=np.eye(4))
    # nib.save(disp_itk_format, f"{footsteps.output_dir}/disp_{fixed_path.split('_')[1]}_{moving_path.split('_')[1]}.nii.gz")

# Prepare submission
import subprocess

#subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_01", f"{footsteps.output_dir}/submission/"])
#subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_02", f"{footsteps.output_dir}/submission/"])
#subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_04", f"{footsteps.output_dir}/submission/"])
subprocess.run(["touch", f"{footsteps.output_dir}/submission/test.txt"])
subprocess.call("zip -r submission.zip ./*", shell=True, cwd=f"{footsteps.output_dir}/submission/")   

logger.log(f"Mean folds percentage: {np.mean(flips)}")
