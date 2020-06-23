import numpy as np

from imgaug import augmenters as iaa


def augment(batch, kinds, random_state):
    if len(batch['image'].shape) != 4:
        return batch
    assert len(batch['image'].shape) == 4
    batch_aug = {}
    batch_aug.update(batch)
    seq = iaa.SomeOf(
        (0, None),
        [augmentations[kind] for kind in kinds],
        random_order=True,
        random_state=random_state,
    )
    batch_aug['image'] = seq.augment_images(batch_aug['image'])
    return batch_aug

augmentations = {
    'fliplr': iaa.Fliplr(0.5),
    'translate_px': iaa.Affine(translate_px={"x": (-4, 4), "y": (-4, 4)}), # CIFAR
    'croppad_px': iaa.CropAndPad(px=((-8,8),(-8,8),(-8,8),(-8,8))), # Tiny-ImageNet
}
