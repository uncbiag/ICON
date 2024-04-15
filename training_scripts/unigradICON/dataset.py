
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fv
import tqdm
import itk
import glob
import monai
from monai.transforms import CropForeground, SpatialPad, ResizeWithPadOrCrop

class HCPDataset(torch.utils.data.Dataset):
    def __init__(self, desired_shape=None) -> None:
        super().__init__()
        with open(f"/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/brain_t1_pipeline/splits/train.txt") as f:
            self.image_paths = f.readlines()
        self.transform = [CropForeground(lambda x: x>0)]
        if desired_shape is not None:
            self.transform.append(SpatialPad(desired_shape))
    
    def __len__(self):
        return len(self.image_paths)

    def process(self, iA, isSeg=False):
        iA = iA[None, None, :, :, :]
        iA = torch.nn.functional.avg_pool3d(iA, 2)[0]
        iA = iA / torch.max(iA)
        for t in self.transform:
            iA = t(iA)
        return iA

    def __getitem__(self, idx):
        img_1 = self.image_paths[idx].split(".nii.gz")[0] + "_restore_brain.nii.gz"
        img_2 = self.image_paths[np.random.randint(0, len(self.image_paths))].split(".nii.gz")[0] + "_restore_brain.nii.gz"
        
        images = []
        for f_name in [img_1, img_2]:
            image = torch.tensor(np.asarray(itk.imread(f_name.replace("playpen-raid2/Data", "playpen-ssd/lin.tian/data_local"))))
            images.append(self.process(image))
        return images
    
class OAIDataset(torch.utils.data.Dataset):
    def __init__(self, desired_shape=None) -> None:
        super().__init__()
        with open(f"/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/oai_paper_pipeline/splits/train/pair_path_list.txt") as f:
            train_pair_paths = f.readlines()
            knee_image_paths_set = set()
            knee_image_paths = []
            for p in train_pair_paths:
                p_s = p.split()
                if p_s[0] not in knee_image_paths_set:
                    knee_image_paths.append([p_s[0], p_s[2]])
                    knee_image_paths_set.add(p_s[0])
                if p_s[1] not in knee_image_paths_set:
                    knee_image_paths.append([p_s[1], p_s[3]])
                    knee_image_paths_set.add(p_s[1])
            self.img_paths = knee_image_paths
        self.transform = None
        if desired_shape is not None:
            self.transform = ResizeWithPadOrCrop(desired_shape)
    
    
    def __len__(self):
        return len(self.img_paths)

    def process(self, iA):
        iA = iA[None, None, :, :, :]
        iA = torch.nn.functional.avg_pool3d(iA, 2)[0]
        iA = self.transform(iA) if self.transform is not None else iA
        return iA

    def __getitem__(self, idx):
        img_1 = self.img_paths[idx]
        img_2 = self.img_paths[np.random.randint(0, len(self.img_paths))]

        images = []
        for f_name in [img_1, img_2]:
            f_img, f_img_seg = [f.replace("playpen/zhenlinx/Data", "playpen-ssd/lin.tian/data_local") for f in f_name]
            img = torch.tensor(np.asarray(itk.imread(f_img)))
            # img_seg = torch.tensor(np.asarray(itk.imread(f_img_seg)))
            if "RIGHT" in f_img:
                img = torch.flip(img, [0])
                # img_seg = torch.flip(img_seg, [0])
            elif "LEFT" in f_img:
                pass
            else:
                raise AssertionError()

            images.append(self.process(img))
        return images

class COPDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ids_file="/playpen-raid2/lin.tian/projects/icon_lung/ICON_lung/splits",
        data_path="/playpen-ssd/lin.tian/data_local/Lung_Registration_transposed/",
        data_num=-1,
        desired_shape=None
    ):
        with open(ids_file) as f:
            self.pair_paths = f.readlines()
            self.pair_paths = list(map(lambda x: x[:-1], self.pair_paths))
        self.data_path = data_path
        self.data_num = data_num
        self.transform = [CropForeground(lambda x: x>0)]
        if desired_shape is not None:
            self.transform.append(SpatialPad(desired_shape))

    def __len__(self):
        return len(self.pair_paths) if self.data_num < 0 else self.data_num

    def process(self, iA, isSeg=False):
        iA = iA[None, None, :, :, :]
        # SI flip
        iA = torch.flip(iA, dims=(2,))
        if isSeg:
            iA = iA.float()
            iA[iA > 0] = 1
            iA = torch.nn.functional.avg_pool3d(iA, 2)
        else:
            iA = iA.float()
            iA = torch.clip(iA, -1000, 0) + 1000.0
            iA = iA / 1000.0
            iA = torch.nn.functional.avg_pool3d(iA, 2)
        return iA

    def __getitem__(self, idx):
        case_id = self.pair_paths[idx]
        image_insp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_INSP_STD*_COPD_img.nii.gz"
                        )[0]
                    )
                )
            )
        )
        image_exp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_EXP_STD*_COPD_img.nii.gz"
                        )[0]
                    )
                )
            )
        )

        seg_insp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_INSP_STD*_COPD_label.nii.gz"
                        )[0]
                    )
                )
            ),
            isSeg=True,
        )
        seg_exp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_EXP_STD*_COPD_label.nii.gz"
                        )[0]
                    )
                )
            ),
            isSeg=True,
        )

        images = [image_insp[0] * seg_insp[0], image_exp[0] * seg_exp[0]]
        for t in self.transform:
            images[0] = t(images[0])
            images[1] = t(images[1])

        return images