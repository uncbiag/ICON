# register_pair.py
import numpy as np

import argparse
import itk
import icon_registration.itk_wrapper
import icon_registration.config
import torch

import icon_registration.unicarl.unigradicon_train_parallel


def get_model(path=None):
    net = icon_registration.unicarl.unigradicon_train_parallel.make_net(3, unigradicon_train_parallel.input_shape, equivariantize=False)
    if path:
        net.regis_net.load_state_dict(torch.load(path))
    net.regis_net = net.regis_net.netPsi.netPsi
    net.assign_identity_map([1, 1, 160, 160, 160])
    net.cuda()
    net.eval()
    return net


def itk_crop_foreground(image, additional_crop_pixels=0):

    def first_nonequal(fn):
        i = 0
        while True:
            slice = fn(i)
            if not np.all(slice == slice.flat[0]):
                return i + additional_crop_pixels
            i += 1

    arr = itk.GetArrayFromImage(image)

    
    if len(arr.shape) == 2:
        
        upper_1 = first_nonequal(lambda i: arr[:, arr.shape[1] - 1 - i])
        upper_2 = first_nonequal(lambda i: arr[arr.shape[0] - 1 - i])
    
        upper = (upper_1, upper_2)
    
        lower_1 = first_nonequal(lambda i: arr[:, i])
        lower_2 = first_nonequal(lambda i: arr[i])
    
        lower = (lower_1, lower_2)
    if len(arr.shape) == 3:
        upper_1 = first_nonequal(lambda i: arr[:, :, arr.shape[2] - 1 - i])
        upper_2 = first_nonequal(lambda i: arr[:, arr.shape[1] - 1 - i])
        upper_3 = first_nonequal(lambda i: arr[arr.shape[0] - 1 - i])
    
        upper = (upper_1, upper_2, upper_3)
        
        lower_1 = first_nonequal(lambda i: arr[:, :, i])
        lower_2 = first_nonequal(lambda i: arr[:, i])
        lower_3 = first_nonequal(lambda i: arr[i])
    
        lower = (lower_1, lower_2, lower_3)

    print(lower, upper)

        
    
    crop_filter = itk.CropImageFilter[type(image), type(image)].New()
    crop_filter.SetUpperBoundaryCropSize(upper)
    crop_filter.SetLowerBoundaryCropSize(lower)
    crop_filter.SetInput(image)
    crop_filter.Update()

    return crop_filter.GetOutput()


def quantile(arr: torch.Tensor, q):
    arr = arr.flatten()
    l = len(arr)
    return torch.kthvalue(arr, int(q * l)).values

def reorient(moving):
        from itk.ITKCommonBasePython import itkSpatialOrientationAdapter

        return itk.orient_image_filter(
                            moving, 
                                    desired_coordinate_orientation=itk.ITKCommonBasePython.itkSpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAS,
                                            use_image_direction=True)

def preprocess(image):
    image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    image = reorient(image)



    image = itk_crop_foreground(image, additional_crop_pixels=9)
    min_ = quantile(torch.tensor(np.array(image)), .01).item()
    max_ = quantile(torch.tensor(np.array(image)), .99).item()
    image = itk.clamp_image_filter(image, Bounds=(min_, max_))

    image = itk.shift_scale_image_filter(image, shift=-min_, scale = 1/(max_-min_)) 

    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register two images")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--transform_out", required=False,
                         default="trans.hdf5", type=str, help="The path to save the transform.")
    parser.add_argument("--warped_moving_out", required=False,
                        default=None, type=str, help="The path to save the warped image.")
    parser.add_argument("--io_iterations", required=False,
                         default="None", help="The number of IO iterations. Default is 50. Set to 'None' to disable IO.")

    args = parser.parse_args()

    net = get_model()

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)

    if args.io_iterations == "None":
        io_iterations = None
    else:
        io_iterations = int(args.io_iterations)

    phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
        net,
        preprocess(moving),
        preprocess(fixed),
        finetune_steps=io_iterations)

    itk.transformwrite([phi_AB], args.transform_out)

    if args.warped_moving_out:
        moving = itk.CastImageFilter[type(moving), itk.Image[itk.F, 3]].New()(moving)
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
        warped_moving_image = itk.resample_image_filter(
                moving,
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed
                )
        itk.imwrite(warped_moving_image, args.warped_moving_out)
    """
    python register.py --fixed register_py/
    """
