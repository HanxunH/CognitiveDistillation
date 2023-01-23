from torchvision import datasets


class CustomCelebA(datasets.CelebA):
    def __init__(self, root='/data/projects/punim0784/datasets',
                 split="train", target_type='attr', transform=None,
                 target_transform=None, download=False, **kwargs):
        super().__init__(root=root, split=split, target_type=target_type,
                         transform=transform, target_transform=target_transform,
                         download=download)
        # attr_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
        # "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
        # "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
        # "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
        # "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
        # "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
        # "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
        # "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
        # "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        # "Wearing_Necklace", "Wearing_Necktie"]

        # Select target attr
        print(self.attr.shape, len(self.filename))
        if 'attr_targets' in kwargs and target_type == 'attr':
            attr_targets = kwargs['attr_targets']
            idx = []
            for attr_target in attr_targets:
                if attr_target in self.attr_names:
                    idx.append(self.attr_names.index(attr_target))
            if len(idx) != 0:
                self.attr = self.attr[:, idx]
                print(self.attr.shape)
