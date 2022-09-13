from torch.utils.data import DataLoader

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as tf


def cityscapes_sequence(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers, window):
    """A loader that loads images for adaptation from the cityscapes_sequence validation set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms = [
        #tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize((resize_height, resize_width), image_types=('color',)),
        tf.ConvertSegmentation(),
        tf.CreateColoraug(new_element=True),
        #tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'cityscapes_sequence_adaptation'),
        tf.AddKeyValue('purposes', ('segmentation',)),
    ]

    dataset_name = 'cityscapes_sequence'

    dataset = StandardDataset(
        dataset=dataset_name,
        trainvaltest_split='validation',
        video_mode='video',
        stereo_mode='mono',
        labels_mode='fromid',
        keys_to_load=('color', 'segmentation'),
        keys_to_video=('color',),
        data_transforms=transforms,
        video_frames=window
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the cityscapes validation set for adaptation", flush=True)

    return loader
