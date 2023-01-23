import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



class FCT_Detection(nn.Module):
    def __init__(self, model, train_loader):
        super(FCT_Detection, self).__init__()
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
            param.requires_grad = True
        self.model.train()
        self.finetune_l_intra()
        self.model.eval()

    def finetune_l_intra(self):
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/finetune_attack_noTrans.py
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.get_features = True
        else:
            self.model.get_features = True

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        for epoch in range(10):
            pbar = tqdm(self.train_loader)
            for images, labels in pbar:
                # Features and Outputs
                images, labels = images.to(device), labels.to(device)
                features, logits = self.model(images)
                features = features[-1]
                # Calculate intra-class loss
                centers = []
                for j in range(logits.shape[1]):
                    j_idx = torch.where(labels == j)[0]
                    if j_idx.shape[0] == 0:
                        continue
                    j_features = features[j_idx]
                    j_center = torch.mean(j_features, dim=0)
                    centers.append(j_center)

                centers = torch.stack(centers, dim=0)
                centers = F.normalize(centers, dim=1)
                similarity_matrix = torch.matmul(centers, centers.T)
                mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(device)
                similarity_matrix[mask] = 0.0
                loss = torch.mean(similarity_matrix)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description("Loss {:.4f}".format(loss.item()))

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.get_features = False
        else:
            self.model.get_features = False


    def transforms(self, images):
        new_imgs = []
        for img in images:
            new_imgs.append(self.transforms_op(img))
        new_imgs = torch.stack(new_imgs)
        return new_imgs


    def forward(self, model, images, labels):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.get_features = True
        else:
            self.model.get_features = True
        
        aug_imgs = self.transforms(images.cpu())
        aug_imgs = aug_imgs.to(device)
        with torch.no_grad():
            # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/calculate_consistency.py
            features1, _ = model(images)
            features2, _ = model(aug_imgs)
            features1 = features1[-1]  # activations of last hidden layer
            features2 = features2[-1]  # activations of last hidden layer
            ### Calculate consistency ###
            feature_consistency = torch.mean((features1 - features2)**2, dim=1)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.get_features = False
        else:
            self.model.get_features = False

        return feature_consistency
