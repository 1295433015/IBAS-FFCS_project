import torch
from monai.metrics import DiceMetric
from monai.transforms import Compose, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import os


def train(model, train_loader, val_loader, optimizer, loss_function, device, root_dir, max_epochs=250, val_interval=10):
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_metric = -1
    best_metric_epoch = -1
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=11)])
    post_label = Compose([AsDiscrete(to_onehot=11)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].round().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        scheduler.step()

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].round().to(device)
                    roi_size = (96, 96, 256) # Modify this parameter when the training objective changes.
                    sw_batch_size = 1
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                print(f"epoch {epoch + 1} validation mean dice: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, f"best_metric_model_epoch_{epoch + 1}.pth"))
                    print("saved new best metric model")