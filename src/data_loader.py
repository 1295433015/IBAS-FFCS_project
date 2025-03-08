import glob
import os
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandFlipd, RandGaussianNoised, RandAdjustContrastd
)
from monai.data import Dataset, DataLoader, CacheDataset


def get_data_dicts(data_dir, data_type="train"):
    images = sorted(glob.glob(os.path.join(data_dir, data_type, "fat", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, data_type, "label", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
    return data_dicts


def get_transforms(train=True):
    if train:
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandGaussianNoised(keys="image", prob=0.5, std=0.01),
                RandAdjustContrastd(keys='image', prob=0.3, gamma=(0.3, 1.3)),
            ]
        )
    else:
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
            ]
        )


def get_dataloaders(train_files, val_files, batch_size=3, num_workers=4):
    train_transforms = get_transforms(train=True)
    val_transforms = get_transforms(train=False)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    return train_loader, val_loader