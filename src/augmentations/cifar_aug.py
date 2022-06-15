import torchvision.transforms as T

class CIFARTransform():
    def __init__(self, normalize, image_size=32, is_train=True, to_pil_image=False):
        self.not_aug_transform = T.Compose([T.ToTensor()])
        if is_train:
            if to_pil_image:
                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.RandomCrop(image_size, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(*normalize)
                ])
            else:
                self.transform = T.Compose([
                    #T.ToPILImage(),
                    T.RandomCrop(image_size, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(*normalize)
                ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(*normalize)
        ])

    def __call__(self, x):        
        aug_x = self.transform(x)
        return aug_x
        