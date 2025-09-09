import paddle
import paddle.nn as nn


class Feature_Detection:
    def __init__(self):
        pass

    def __call__(self, model, images, labels):
        if isinstance(model, paddle.DataParallel):
            raise NotImplementedError("Skip")
        else:
            model.get_features = True
        with paddle.no_grad():
            features, _ = model(images)
            features = features[-1]  # activations of last hidden layer
        if isinstance(model, paddle.DataParallel):
            raise NotImplementedError("Skip")
        else:
            model.get_features = False
        return features