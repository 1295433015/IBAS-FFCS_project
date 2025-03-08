import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet
from monai.networks.layers import Norm


def load_model(checkpoint_path, device):
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=11,
        norm=Norm.BATCH,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def run_inference(data_root, save_root, model, device):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    test_files = [{"image": os.path.join(data_root, file)} for file in os.listdir(data_root)]
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        ]
    )
    dataset = Dataset(data=test_files, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, t_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            roi_size = (96, 96, 256)
            sw_batch_size = 1
            test_outputs = sliding_window_inference(t_data["image"].to(device), roi_size, sw_batch_size, model)
            test_outputs = torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, :]
            test = np.array(test_outputs, dtype=np.uint8)

            fat = nib.load(test_files[i]['image'])
            pred_nib = nib.Nifti1Image(test, fat.affine, fat.header)
            nib.save(pred_nib, os.path.join(save_root, os.path.basename(test_files[i]['image'])))


if __name__ == '__main__':
    data_root = '/path/to/data_root'
    save_root = '/path/to/save_root'
    checkpoint_path = '/path/to/checkpoint.pth'
    device = torch.device("cuda:1")

    model = load_model(checkpoint_path, device)
    run_inference(data_root, save_root, model, device)