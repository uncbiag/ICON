
import numpy as np
import itk
import tqdm
import torch

from monai.transforms import CropForeground, SpatialPad, ResizeWithPadOrCrop, SpatialCrop
with open(f"/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/brain_t1_pipeline/splits/train.txt") as f:
    image_paths = f.readlines()
f_path = image_paths[0].split(".nii.gz")[0] + "_restore_brain.nii.gz"
import ITK_transforms
import importlib
def test_itk_crop_foreground():
    importlib.reload(ITK_transforms)

    # crop foreground test
    image = itk.imread(f_path)

    print("Original shape:", np.asarray(image).shape)
    transform = CropForeground(lambda x: x>0)

    cropped_monai = transform(torch.tensor(np.asarray(image))[None, None])

    print("Cropped monai shape", cropped_monai.shape)
    cropped_itk =torch.tensor( np.asarray(ITK_transforms.itk_crop_foreground(image, transform)))[None, None]

    print("itk, monai:", cropped_itk.shape, cropped_monai.shape)


    print(torch.sum((cropped_monai - cropped_itk)**2))

def test_itk_crop_filter():
    
    importlib.reload(ITK_transforms)

    image = itk.imread(f_path)

    transform = ResizeWithPadOrCrop([175, 175, 175]).cropper

    print(transform)

    image_t = torch.tensor(np.asarray(image))[None]

    import pdb
#    pdb.set_trace()
    cropped_monai = transform(image_t)

    print("Cropped monai shape", cropped_monai.shape)

    crop = transform.compute_slices(spatial_size=image_t.shape[1:])

    print(dir(crop[0]))

    


    cropped_itk =torch.tensor( np.asarray(ITK_transforms.itk_crop(image, transform)))[None, None]

    print("itk, monai:", cropped_itk.shape, cropped_monai.shape)


    print(torch.sum((cropped_monai - cropped_itk)**2))

def test_itk_pad_image_filter():
    importlib.reload(ITK_transforms)

    image = itk.imread(f_path)

    image = ITK_transforms.itk_crop_foreground(image, CropForeground())

    tranform = SpatialPad([175, 175, 175])





