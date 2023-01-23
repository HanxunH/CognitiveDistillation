import torch
import torch.nn as nn


class Feature_Detection(nn.Module):
    def __init__(self):
        super(Feature_Detection, self).__init__()
        # Feature extraction for detections

    def forward(self, model, images, labels):
        if isinstance(model, torch.nn.DataParallel):
            model.module.get_features = True
        else:
            model.get_features = True
        with torch.no_grad():
            features, _ = model(images)
            features = features[-1]  # activations of last hidden layer
        if isinstance(model, torch.nn.DataParallel):
            model.module.get_features = False
        else:
            model.get_features = False
        return features
