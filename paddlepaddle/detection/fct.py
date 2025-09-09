import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


class FCT_Detection:
    def __init__(self, model, train_loader):
        # Feature consistency towards transformations
        self.model = model 
        self.train_loader = train_loader
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/utils/dataloader_bd.py
        self.transforms_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
        ])
        # Finetune with L_intra
        for param in self.model.parameters():
            param.stop_gradient = False
        self.model.train()
        self.finetune_l_intra()
        self.model.eval()

    def finetune_l_intra(self):
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/finetune_attack_noTrans.py
        self.model.get_features = True

        optimizer = paddle.optimizer.Momentum(parameters=self.model.parameters(),
                                           learning_rate=0.01,
                                           momentum=0.9,
                                           weight_decay=5e-4)

        for epoch in range(10):
            pbar = tqdm(self.train_loader)
            for images, labels in pbar:
                # Features and Outputs
                images = paddle.to_tensor(images, dtype='float32')
                labels = paddle.to_tensor(labels, dtype='int64')
                features, logits = self.model(images)
                features = features[-1]
                # Calculate intra-class loss
                centers = []
                for j in range(logits.shape[1]):
                    j_idx = paddle.where(labels == j)[0]
                    if j_idx.shape[0] == 0:
                        continue
                    # Use gather to avoid extra dimension issue (see paddlepaddle-exemptions.md #13)
                    j_features = paddle.gather(features, j_idx[:, 0], axis=0)
                    j_center = paddle.mean(j_features, axis=0)
                    centers.append(j_center)

                centers = paddle.stack(centers, axis=0)
                centers = F.normalize(centers, axis=1)
                similarity_matrix = paddle.matmul(centers, paddle.transpose(centers, [1, 0]))
                mask = paddle.eye(similarity_matrix.shape[0]).astype('bool')
                similarity_matrix[mask] = 0.0
                loss = paddle.mean(similarity_matrix)
                
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description("Loss {:.4f}".format(loss.item()))

        self.model.get_features = False


    def transforms(self, images):
        new_imgs = []
        for img in images:
            # Convert PaddlePaddle tensor to numpy for torchvision transforms
            if isinstance(img, paddle.Tensor):
                img_np = img.numpy()
                # Transpose from CHW to HWC for PIL
                if len(img_np.shape) == 3 and img_np.shape[0] <= 4:
                    img_np = img_np.transpose(1, 2, 0)
            else:
                img_np = img
            transformed = self.transforms_op(img_np)
            # Convert torch.Tensor back to paddle.Tensor
            if hasattr(transformed, 'numpy'):  # It's a torch.Tensor
                transformed = paddle.to_tensor(transformed.numpy())
            new_imgs.append(transformed)
        new_imgs = paddle.stack(new_imgs)
        return new_imgs


    def __call__(self, model, images, labels):
        self.model.get_features = True
        
        aug_imgs = self.transforms(images)
        with paddle.no_grad():
            # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/calculate_consistency.py
            features1, _ = model(images)
            features2, _ = model(aug_imgs)
            features1 = features1[-1]  # activations of last hidden layer
            features2 = features2[-1]  # activations of last hidden layer
            ### Calculate consistency ###
            feature_consistency = paddle.mean((features1 - features2)**2, axis=1)

        self.model.get_features = False

        return feature_consistency