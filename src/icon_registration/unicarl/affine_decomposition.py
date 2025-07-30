import itk

import matplotlib.pyplot as plt
import numpy as np
import icon_registration.itk_wrapper

voxels = 160

def decompose_icon_itk_transform(phi_AB:itk.CompositeTransform):

    original_displacement_transform = phi_AB.GetNthTransform(1)
    original_displacement_transform = itk.DisplacementFieldTransform[itk.D, 3].cast(original_displacement_transform)
    displacement_image = original_displacement_transform.GetDisplacementField()
    original_displacement_array = itk.GetArrayFromImage(displacement_image)

    coordinates = np.mgrid[0:voxels, 0:voxels, 0:voxels]
    coordinates = coordinates.transpose((3, 2, 1, 0))
    coordinates = np.concatenate([coordinates, np.ones((voxels, voxels, voxels, 1))], axis=-1)

    x = coordinates.reshape(-1, 4)
    y = original_displacement_array.reshape(-1, 3)

    best_affine_fit = np.linalg.inv(x.T @ x) @ (x.T @ y)

    error =  (y - x @ best_affine_fit)

    Offset = best_affine_fit[3]

    Matrix = best_affine_fit[:3].transpose() + np.eye(3)


    error = error @ np.linalg.inv(Matrix.transpose())
    error = error.reshape(voxels, voxels, voxels, 3)
    residual_displacement_transform = itk.DisplacementFieldTransform[(itk.D, 3)].New()
    itk_disp_field = itk.image_from_array(error, is_vector=True)
    residual_displacement_transform.SetDisplacementField(itk_disp_field)

    transformType = itk.CenteredAffineTransform[itk.D, 3]
    affine_component_of_network_transform = transformType.New()
    affine_component_of_network_transform.SetOffset(Offset)
    affine_component_of_network_transform.SetCenter((0, 0, 0))
    affine_component_of_network_transform.SetMatrix(itk.matrix_from_array(Matrix))

    affine_decomposed_transform = itk.CompositeTransform[itk.D, 3].New()

    affine_decomposed_transform.PrependTransform(phi_AB.GetNthTransform(2)) 
    affine_decomposed_transform.PrependTransform(residual_displacement_transform)
    affine_decomposed_transform.PrependTransform(affine_component_of_network_transform)
    affine_decomposed_transform.PrependTransform(phi_AB.GetNthTransform(0))
    return affine_decomposed_transform

def extract_affine_icon_itk_transform(phi_AB):
    phi_AB = decompose_icon_itk_transform(phi_AB)

    affine_Transform = itk.CompositeTransform[itk.D, 3].New()

    affine_Transform.PrependTransform(phi_AB.GetNthTransform(3))
    affine_Transform.PrependTransform(phi_AB.GetNthTransform(1))
    affine_Transform.PrependTransform(phi_AB.GetNthTransform(0))

    return affine_Transform



