import torch
from monai.networks.nets import BasicUNet
from monai.networks.layers import Norm

def get_model(device, in_channels=1, out_channels=11):
    model = BasicUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        norm=Norm.BATCH,
    ).to(device)
    return model

def get_loss_function():
    from monai.losses import DiceCELoss
    return DiceCELoss(softmax=True, to_onehot_y=True)

def get_optimizer(model, lr=1e-3, weight_decay=1e-5):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, T_max=250):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    return CosineAnnealingLR(optimizer, T_max=T_max)