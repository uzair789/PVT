from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from .dataset import *
from .auto_augment import ImageNetPolicy, Cutout, RandomResize, HueNormalize, fangyi_sc_transform
import numpy as np



class AliProductDataLoader(DataLoader):
    def __init__(self, data_dir, data_list, batch_size, image_size, sample='instance', cutout=False, auto_augment=False,
                 tta=False, hue=False, down_sample=False, warp=False, num_workers=0):
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample = sample
        self.num_workers = num_workers
        self.tta = tta
        self.hue = hue

        if image_size == 224:
            resize_size = 256
        elif image_size == 288:
            resize_size = 300
        else:
            resize_size = 512

        if 'train' in self.data_list:
            Dataset = AliProductDataset
            # Dataset = SCProductDataset
            self.transform = [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]

            if self.hue:
                self.transform.append(HueNormalize(mean=[0.485, 0.456, 0.406, 0.5],
                                                   std=[0.229, 0.224, 0.225, 1.0]))
            else:
                self.transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std = [0.229, 0.224, 0.225]))
            if auto_augment:
                self.transform.insert(1, ImageNetPolicy())
            if down_sample:
                self.transform.insert(1, RandomResize(scale=(0.2, 0.5), prob=0.8))
            if warp:
                self.transform.insert(0, fangyi_sc_transform())
            if cutout:
                self.transform.insert(len(self.transform) - 2, Cutout(32))
            self.transform = transforms.Compose(self.transform)
        elif 'valid' in self.data_list:
            Dataset = AliProductDataset
            # Dataset = SCProductDataset
            if self.tta:
                self.transform = [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(resize_size),
                    transforms.RandomCrop(self.image_size),
                    transforms.ToTensor(),
                ]
            else:
                self.transform = [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                ]
            if self.hue:
                self.transform.append(HueNormalize(mean=[0.485, 0.456, 0.406, 0.5],
                                                   std=[0.229, 0.224, 0.225, 1.0]))
            else:
                self.transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std = [0.229, 0.224, 0.225]))
            if down_sample:
                self.transform.insert(1, RandomResize(scale=(0.2, 0.5), prob=1))
            if warp:
                self.transform.insert(0, fangyi_sc_transform())
            self.transform = transforms.Compose(self.transform)
        else:
            # Dataset = AliProductTestDataset
            Dataset = SCProductTestDataset
            if self.tta:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(resize_size),
                    transforms.RandomCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        self.dataset = Dataset(self.data_dir, self.data_list, self.transform, self.tta)
        num_samples = len(self.dataset)

        # Sample
        if 'train' in data_list:
            if sample == 'balance':
                weights = self.dataset.get_weights()
                sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples)
                super(AliProductDataLoader, self).__init__(
                    dataset=self.dataset,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    drop_last=True,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=self.num_workers)
            elif sample == 'instance':
                super(AliProductDataLoader, self).__init__(
                    dataset=self.dataset,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    drop_last=True,
                    shuffle=True,
                    num_workers=self.num_workers)
        else:
            super(AliProductDataLoader, self).__init__(
                dataset=self.dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers)


if __name__ == '__main__':
    from tqdm import tqdm

    data_dir = '/media/Zeus/Plugs/AliProducts/data_haoc/'
    data_list = '/home/sreena/haoc/Projects/AliProductCls/datasets/list/new_valid_list.txt'
    data_loader = AliProductDataLoader(data_dir, data_list, batch_size=16, image_size=224, sample='instance', num_workers=0, tta=True)

    for idx, (images, targets, mage_paths) in tqdm(enumerate(data_loader)):
        # print(image_paths)
        continue