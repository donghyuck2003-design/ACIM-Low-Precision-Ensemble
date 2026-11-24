import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_cifar10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_cifar100(batch_size, data_root='/home/ssdl_neurosim1/All_Neurosim_Project/VGG8_CIFAR100/CIFAR100_Data', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

import os, glob, re
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageNetValFlatById(Dataset):
    """
    val 폴더에 클래스 폴더가 없고 파일만 있을 때 사용.
    파일명: ILSVRC2012_val_00000001.JPEG
    라벨: 00000001 (문자열) -> class_to_idx로 정수 인덱스 변환
    """
    def __init__(self, val_root, transform=None, pattern=r"ILSVRC2012_val_(\d+)\.JPEG"):
        self.val_root = val_root
        self.transform = transform
        self.regex = re.compile(pattern)

        # 이미지 파일 목록
        self.samples = sorted(
            glob.glob(os.path.join(val_root, "*.JPEG")) +
            glob.glob(os.path.join(val_root, "*.jpg")) +
            glob.glob(os.path.join(val_root, "*.jpeg")) +
            glob.glob(os.path.join(val_root, "*.png"))
        )
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {val_root}")

        # 파일명에서 id 추출 -> classes 구성
        ids = []
        bad = 0
        for p in self.samples:
            m = self.regex.search(os.path.basename(p))
            if m is None:
                bad += 1
                continue
            ids.append(m.group(1))  # 예: "00000001"
        if bad > 0:
            raise RuntimeError(f"{bad} files do not match pattern: {pattern}")

        self.classes = sorted(set(ids))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        fname = os.path.basename(path)
        m = self.regex.search(fname)
        cls_name = m.group(1)  # "00000001"
        target = self.class_to_idx[cls_name]

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def get_imagenet(batch_size,
                 data_root='/ssdl/neurosim_data/ImageNet/extracted_data/ILSVRC/Data/CLS-LOC',
                 train=True, val=True, test=True,
                 val_ratio=0.1,
                 **kwargs):

    import os
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    num_workers = kwargs.get('num_workers', 4)
    pin_memory = kwargs.get('pin_memory', True)

    ds = []

    # -------- train/val: train 폴더에서 split --------
    if train or val:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        train_path = os.path.join(data_root, 'train')
        full_trainset = datasets.ImageFolder(train_path, transform=train_transform)

        val_size = int(len(full_trainset) * val_ratio)
        train_size = len(full_trainset) - val_size
        train_set, val_set = random_split(full_trainset, [train_size, val_size])

        if train:
            train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory
            )
            ds.append(train_loader)

        if val:
            val_loader = DataLoader(
                val_set, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )
            ds.append(val_loader)

    # -------- test: val 폴더(flat)에서 파일명 번호로 라벨링 --------
    if test:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        val_root = os.path.join(data_root, 'val')
        imagenet_testset = ImageNetValFlatById(val_root, transform=test_transform)

        test_loader = DataLoader(
            imagenet_testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        ds.append(test_loader)

    return ds if len(ds) > 1 else ds[0]
