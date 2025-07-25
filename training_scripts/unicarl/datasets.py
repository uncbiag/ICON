import icon_registration.unicarl.dataset as dataset
import torch
import footsteps
import matplotlib.pyplot as plt

# If you are in uncbiag running this script to preprocess the unicarl Dataset from scratch
# First, sorry.
# Second:
# run it on biag-w05 where the autoPET dataset is 
# do a sshfs biag-lambda2:/data /data to access the abdomen8k data on that maching

input_shape = (1, 1, 160, 160, 160)

public = False


datasets_ = []

#datasets = lambda: None
#datasets.append = lambda x: None

maximum_images=1000

cache_filename = "results/unicarl_private/"

datasets_.append(dataset.PairedDataset(input_shape, "HAN-Seg", 
    "/playpen-raid1/Data/HaN-Seg/HaN-Seg/set_1/case_??/case_??_IMG_*.nrrd", match_regex=r"/(case_[0-9]*)/", maximum_images=maximum_images, cache_filename=cache_filename) )
datasets_.append(dataset.DiffusionDataset(          input_shape, "ebrahim-diffusion", "/playpen-raid1/tgreer/ebrahim_brains/data/degree_powers_normalized_dipy/degree_power_images/*", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.PairedDICOMDataset(input_shape, "CPTAC-UCEC", "/playpen-raid1/Data/TCIA_CPTAC-UCEC/manifest-1712342731330/CPTAC-UCEC/*/*/*/", match_regex=r"/(C3[NL]-[0-9]*)/", maximum_images=maximum_images * 4, cache_filename=cache_filename) )
datasets_.append(dataset.PairedDICOMDataset(input_shape, "TCIA-hastings", "/playpen-raid1/Data/TCIA_Hastings_custom_mrct/manifest-1743108366953/*/*/*/*/", match_regex=r"66953/[a-zA-Z\-]*/([A-Z0-9\-]+)/", maximum_images=maximum_images * 4, cache_filename=cache_filename))
datasets_.append(dataset.PairedDICOMDataset(input_shape, "CPTAC-Sarcoma", 
    "/playpen-raid1/Data/TCIA-Sarcoma/manifest-MjbMt99Q1553106146386120388/Soft-tissue-Sarcoma/*/*/?.*/", match_regex=r"/(STS_[0-9]*)/", maximum_images=maximum_images*4, cache_filename=cache_filename) )
if (not public):
    datasets_.append(dataset.PairedDataset(     input_shape, "pancreas", "/playpen-raid1/tgreer/pancreatic_cancer_registration/data/*/Processed/*/original_image.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, match_regex=r"data/([0-9]+)/Processed/"))
    datasets_.append(dataset.PairedDataset(     input_shape, "dirlab_clamped", "/playpen-raid2/Data/Lung_Registration_clamp_normal_transposed/*/*_img.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, match_regex=r'transposed/([a-zA-Z0-9]+)/'))
    datasets_.append(dataset.PairedDataset(     input_shape, "dirlab", "/playpen-raid2/Data/Lung_Registration_transposed/*/*_img.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, match_regex=r'transposed/([a-zA-Z0-9]+)/'))
    datasets_.append(dataset.Dataset(           input_shape, "HCP_t1_stripped", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T1w_acpc_dc_restore_brain.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
    datasets_.append(dataset.Dataset(           input_shape, "HCP_t1", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T1w_acpc_dc_restore.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
    datasets_.append(dataset.Dataset(           input_shape, "HCP_t2_stripped", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T2w_acpc_dc_restore_brain.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
    datasets_.append(dataset.Dataset(           input_shape, "HCP_t2", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T2w_acpc_dc_restore.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
    datasets_.append(dataset.Dataset(           input_shape, "OAI", "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled_LEFT/*_image.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
    datasets_.append(dataset.Dataset(           input_shape, "translucence", "/playpen-raid1/tgreer/mouse_brain_translucence/data/auto_files_resampled/*", cache_filename=cache_filename, maximum_images=maximum_images))
datasets_.append(dataset.Dataset(           input_shape, "abdomen8k", "/data/hastings/Abdomen8k/AbdomenAtlas/*/ct.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, shuffle=True))
datasets_.append(dataset.PairedDICOMDataset(input_shape, "DukeLivers", "/playpen-raid1/Data/DukeLivers/Segmentation/Segmentation/*/*/images/", maximum_images=maximum_images, match_regex=r"Segmentation/([0-9]+)/", cache_filename=cache_filename))
#datasets_.append(dataset.Dataset(           input_shape, "TotalSegmentatorMRI", "/playpen-raid1/soumitri/data/TotalSegMRI/*/*/mri.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.PairedDataset(     input_shape, "bratsreg", "/playpen-raid2/Data/BraTS-Reg/BraTSReg_Training_Data_v3/*/*.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, match_regex=r"v3/(BraTSReg_[0-9]+)/"))
datasets_.append(dataset.Dataset(           input_shape, "abdomen1k", "/playpen-raid2/Data/AbdomenCT-1K/AbdomenCT-1K-ImagePart*/Case_*", maximum_images=maximum_images, cache_filename=cache_filename, shuffle=True))
datasets_.append(dataset.Dataset(           input_shape, "fmost", "/playpen-raid2/Data/fMost/subject/*_red_mm_RSA.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(           input_shape, "oasis", "/playpen-raid2/Data/oasis/OASIS_OAS1_*_MR1/orig.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(           input_shape, "lumir", "/playpen-raid1/Data/LUMIR/imagesTr/*", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.PairedDataset(     input_shape, "autoPET", "/playpen1/tgreer/PET/FDG-PET-CT-Lesions/*/*/[PC][TE][Tr]*.nii.gz", match_regex=r"(/PETCT_[0-9a-z]+/)", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.PairedDataset(input_shape, "anatomix", 
    "/playpen-raid1/tgreer/anatomix/anatomix/synthetic-data-generation/synthesized_views/view*/*Z.nii.gz", match_regex=r"([0-9A-Z]+).nii.gz", maximum_images=8 * maximum_images, cache_filename=cache_filename) )

import torch.nn.functional as F

def zoom_3d_image(image, zoom):
   """zoom: numpy array [zoom_x, zoom_y, zoom_z]"""
   #print(image.shape)
   batch_size = image.shape[0]
   
   theta = torch.tensor([
       [zoom[2], 0, 0, 0],
       [0, zoom[1], 0, 0], 
       [0, 0, zoom[0], 0]
   ], dtype=image.dtype, device=image.device).unsqueeze(0).repeat(batch_size, 1, 1)
   
   grid = F.affine_grid(theta, image.size(), align_corners=False)
   #print(grid.shape)
   return F.grid_sample(image, grid, align_corners=False, mode='bilinear')


if __name__ == "__main__":
    for d in datasets_:
        for i in range(5):
            
            meta_pair = d.get_pair()
            pair = [meta_pair[0][0], meta_pair[1][0]]
            spacing_ratio = meta_pair[0][1] / meta_pair[1][1]

            pair[1] = zoom_3d_image(pair[1], spacing_ratio)
        

            plt.imshow(torch.cat(pair, dim=2)[0, 0, :, :, 50].cpu())
            footsteps.plot("-_____________________________" + d.name)
            #plt.imshow(torch.max(torch.cat(pair, dim=2), dim=4).values[0, 0].cpu())
            #footsteps.plot(d.name)
            #plt.imshow(torch.cat(pair, dim=2)[0, 0, :, 50].cpu())
            #footsteps.plot(d.name)
            #plt.imshow(torch.cat(pair, dim=4)[0, 0, 50].cpu())
            #footsteps.plot(d.name)
