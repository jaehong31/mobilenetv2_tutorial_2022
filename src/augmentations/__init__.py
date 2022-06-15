from .cifar_aug import CIFARTransform
cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]

def get_aug(name='cifar', is_train=True, to_pil_image=False):
    if name == 'cifar':
        augmentation = CIFARTransform(normalize=cifar_norm, is_train=is_train, to_pil_image=to_pil_image)
    else:
        raise NotImplementedError
    return augmentation
