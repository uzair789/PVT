import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
__all__ = ['AliProductDataset', 'AliProductTestDataset', 'SCProductDataset', 'SCProductTestDataset']


class AliProductDataset(Dataset):
    def __init__(self, data_dir, data_list, transform=None, tta=False):
        self.data_dir = data_dir
        self.data, self.labels = self._load_data(data_list)
        self.transform = transform
        self.num_classes = len(set(self.labels))
        self.tta = tta
        print("total {} images, {} classes".format(len(self.data), self.num_classes))


    @staticmethod
    def _load_data(data_list):
        data, label = [], []
        with open(data_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                img_path = line[0]
                class_id = line[1]
                data.append(img_path)
                label.append(int(class_id))
        print('classes in load method', len(label))
        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        while True:
            try:
                image = Image.open(self.data_dir + image_path)
                image = image.convert('RGB')
                break
            except:
                image_path = self.data[np.random.randint(0, len(self.data))]
        label = self.labels[index]
        label = torch.tensor(label).long()

        if self.tta:
            images = []
            for i in range(5):
                images.append(self.transform(image))
            image = np.stack(images)
        else:
            if self.transform:
                image = self.transform(image)

        return image, label#, image_path

    def get_weights(self):
        class_dict = {k: [] for k in range(self.num_classes)}
        class_weights = {k: 0 for k in range(self.num_classes)}

        for data, label in zip(self.data, self.labels):
            class_dict[label].append(str(data))

        total_images = len(self.data)
        for label, weight in class_weights.items():
            num_images = len(class_dict[label])
            if num_images == 0:
                class_weights[label] = 0
            else:
                class_weights[label] = total_images / num_images

        weights = []
        for label in self.labels:
            weights.append(class_weights[label])

        return weights


class AliProductTestDataset(Dataset):
    def __init__(self, data_dir, data_list, transform=None, tta=False):
        assert 'test_list' in data_list, "test data list {} not supported yet".format(data_list)
        self.data_dir = data_dir
        self.data = self._load_data(data_list)
        self.transform = transform
        self.tta = tta

    @staticmethod
    def _load_data(data_list):
        data = []
        with open(data_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                img_path = line[0]
                data.append(img_path)
        print("{} images".format(len(data)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        image_name = image_path.split('/')[-1]
        image = Image.open(self.data_dir + image_path)
        image = image.convert('RGB')

        if self.tta:
            images = []
            for i in range(5):
                images.append(self.transform(image))
            image = np.stack(images)
            return image, image_name
        else:
            if self.transform:
                image = self.transform(image)
            return image, image_name

class SCProductDataset(Dataset):
    def __init__(self, data_dir, data_list, transform=None, tta=False):
        self.data_dir = data_dir
        self.data, self.labels, self.bbox = self._load_data(data_list)
        self.transform = transform
        self.num_classes = len(set(self.labels))
        self.tta = tta
        print("total {} images, {} classes".format(len(self.data), self.num_classes))

    @staticmethod
    def _load_data(data_list):
        data, label, bbox = [], [], []
        dl_json = json.load(open(data_list, "r"))
        for k, l, v in dl_json:
            data += [k]
            label += [l]
            bbox += [v]
        return data, label, bbox

    def __len__(self):
        return len(self.data) * len(self.data_dir)

    def __getitem__(self, index):
        while True:
            dir_set = index // len(self.data)
            index = index % len(self.data)
            image_path = self.data[index]
            full_path = self.data_dir[dir_set] + image_path
            if os.path.exists(full_path):
                image = Image.open(full_path)
                break
            else:
                print(f'missing image: {self.data_dir[dir_set] + image_path}')
                index = np.random.randint(0, self.__len__())
        image = image.convert('RGB')
        image = image.crop(self.bbox[index])
        label = self.labels[index]
        label = torch.tensor(label).long()

        if self.tta:
            images = []
            for i in range(5):
                images.append(self.transform(image))
            image = np.stack(images)
        else:
            if self.transform:
                image = self.transform(image)

        return image, label, image_path

    def get_weights(self):
        class_dict = {k: [] for k in range(self.num_classes)}
        class_weights = {k: 0 for k in range(self.num_classes)}

        for data, label in zip(self.data, self.labels):
            class_dict[label].append(str(data))

        total_images = len(self.data)
        for label, weight in class_weights.items():
            num_images = len(class_dict[label])
            if num_images == 0:
                class_weights[label] = 0
            else:
                class_weights[label] = total_images / num_images

        weights = []
        for label in self.labels:
            weights.append(class_weights[label])

        return weights


class SCProductTestDataset(Dataset):
    def __init__(self, data_dir, data_list, transform=None, tta=False):
        assert 'test_list' in data_list, "test data list {} not supported yet".format(data_list)
        self.data_dir = data_dir
        self.data = self._load_data(data_list)
        self.transform = transform
        self.tta = tta

    @staticmethod
    def _load_data(data_list):
        data = []
        with open(data_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                img_path = line[0]
                data.append(img_path)
        print("{} images".format(len(data)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        image_name = image_path.split('/')[-1]
        image = Image.open(self.data_dir + image_path)
        image = image.convert('RGB')

        if self.tta:
            images = []
            for i in range(5):
                images.append(self.transform(image))
            image = np.stack(images)
            return image, image_name
        else:
            if self.transform:
                image = self.transform(image)
            return image, image_name


if __name__ == '__main__':
    from tqdm import tqdm

    data_dir = '/media/Zeus/Plugs/AliProducts/data_haoc'
    data_list = '/home/sreena/haoc/Projects/AliProductCls/datasets/list/test_list.txt'
    train_dataset = AliProductTestDataset(data_dir, data_list, tta=True)

    # check all images
    for i in tqdm(range(len(train_dataset))):
        data = train_dataset.__getitem__(i)

