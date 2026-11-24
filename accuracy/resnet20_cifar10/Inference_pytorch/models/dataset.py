import os
import csv
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

# ----------------------------
# CIFAR10 / CIFAR100 (그대로)
# ----------------------------
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


def get_cifar100(batch_size, data_root='/home/ssdl_neurosim1/All_Neurosim_Project/VGG8_CIFAR100/Data',
                 train=True, val=True, **kwargs):
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


# ----------------------------
# ImageNet LOC 파일 기반 라벨링
# ----------------------------
def build_wnid_to_idx(synset_mapping_path):
    """LOC_synset_mapping.txt의 줄 번호(0-based)를 class index로 사용"""
    wnid_to_idx = {}
    with open(synset_mapping_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            wnid = line.split()[0]
            wnid_to_idx[wnid] = i
    return wnid_to_idx


def load_imageid_to_label(solution_csv_path, wnid_to_idx):
    """
    LOC_*_solution.csv에서 ImageId -> class_idx 매핑 생성
    PredictionString의 첫 토큰(wnid)만 사용 (classification 라벨)
    """
    imageid_to_label = {}
    with open(solution_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageId"]
            pred = row["PredictionString"].strip()
            if not pred:
                continue
            wnid = pred.split()[0]  # 첫 wnid
            if wnid not in wnid_to_idx:
                continue
            imageid_to_label[image_id] = wnid_to_idx[wnid]
    return imageid_to_label


class ImageNetTrainFromLocSolution(Dataset):
    """
    train 이미지: {train_root}/{wnid}/{ImageId}.JPEG
    라벨: LOC_train_solution.csv + synset mapping으로 0~999
    """
    def __init__(self, train_root, train_solution_csv, wnid_to_idx, transform=None, ext=".JPEG"):
        self.train_root = train_root
        self.transform = transform
        self.ext = ext

        self.imageid_to_label = load_imageid_to_label(train_solution_csv, wnid_to_idx)

        # solution.csv에 있는 ImageId 목록만 사용
        self.ids = list(self.imageid_to_label.keys())
        if len(self.ids) == 0:
            raise RuntimeError(f"No ids loaded from {train_solution_csv}")

        # ImageId -> wnid도 필요(경로 구성 위해)
        # wnid는 PredictionString 첫 토큰
        # 다시 한 번 읽되, 메모리 절약 위해 필요한 것만 생성
        self.imageid_to_wnid = {}
        with open(train_solution_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["ImageId"]
                pred = row["PredictionString"].strip()
                if not pred:
                    continue
                wnid = pred.split()[0]
                if image_id in self.imageid_to_label:
                    self.imageid_to_wnid[image_id] = wnid

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        target = self.imageid_to_label[image_id]
        wnid = self.imageid_to_wnid[image_id]
        path = os.path.join(self.train_root, wnid, image_id + self.ext)

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


class ImageNetValFromLocSolution(Dataset):
    """
    val 이미지: {val_root}/{ImageId}.JPEG   (ImageId = ILSVRC2012_val_000xxxxx)
    라벨: LOC_val_solution.csv + synset mapping으로 0~999
    """
    def __init__(self, val_root, val_solution_csv, wnid_to_idx, transform=None, ext=".JPEG"):
        self.val_root = val_root
        self.transform = transform
        self.ext = ext

        self.imageid_to_label = load_imageid_to_label(val_solution_csv, wnid_to_idx)
        self.ids = list(self.imageid_to_label.keys())
        if len(self.ids) == 0:
            raise RuntimeError(f"No ids loaded from {val_solution_csv}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        target = self.imageid_to_label[image_id]
        path = os.path.join(self.val_root, image_id + self.ext)

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


def get_imagenet(batch_size,
                 data_root='/ssdl/neurosim_data/ImageNet/extracted_data/ILSVRC',  # ILSVRC 루트로 받음
                 loc_root='/ssdl/neurosim_data/ImageNet/extracted_data',
                 val_ratio=0.1,
                 **kwargs):


    num_workers = kwargs.get('num_workers', 4)
    pin_memory = kwargs.get('pin_memory', True)

    # ✅ data_root가 ILSVRC 루트인지, CLS-LOC 루트인지 자동 판별
    cand1 = os.path.join(data_root, "Data/CLS-LOC")   # ILSVRC 루트일 때
    cand2 = data_root                                # 이미 CLS-LOC일 때

    if os.path.isdir(os.path.join(cand1, "train")) and os.path.isdir(os.path.join(cand1, "val")):
        cls_loc_root = cand1
    elif os.path.isdir(os.path.join(cand2, "train")) and os.path.isdir(os.path.join(cand2, "val")):
        cls_loc_root = cand2
    else:
        raise RuntimeError(f"Cannot find CLS-LOC train/val under: {data_root}")

    train_root = os.path.join(cls_loc_root, "train")
    val_root   = os.path.join(cls_loc_root, "val")

    # transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    synset_path = os.path.join(loc_root, "LOC_synset_mapping.txt")
    train_sol   = os.path.join(loc_root, "LOC_train_solution.csv")
    val_sol     = os.path.join(loc_root, "LOC_val_solution.csv")

    wnid_to_idx = build_wnid_to_idx(synset_path)

    full_trainset = ImageNetTrainFromLocSolution(
        train_root=train_root,
        train_solution_csv=train_sol,
        wnid_to_idx=wnid_to_idx,
        transform=train_transform,
        ext=".JPEG"
    )

    val_size = int(len(full_trainset) * val_ratio)
    train_size = len(full_trainset) - val_size
    train_set, val_set = random_split(full_trainset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    test_set = ImageNetValFromLocSolution(
        val_root=val_root,
        val_solution_csv=val_sol,
        wnid_to_idx=wnid_to_idx,
        transform=test_transform,
        ext=".JPEG"
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader