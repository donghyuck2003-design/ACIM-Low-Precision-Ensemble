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
    def __init__(self, train_root, train_solution_csv, wnid_to_idx,
                 transform=None, ext=".JPEG", check_exists=True):
        self.train_root = train_root
        self.transform = transform
        self.ext = ext
        self.samples = []

        with open(train_solution_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["ImageId"]
                pred = row["PredictionString"].strip()
                if not pred:
                    continue

                wnid = pred.split()[0]  # class wnid
                if wnid not in wnid_to_idx:
                    continue

                path = os.path.join(train_root, wnid, image_id + ext)
                if check_exists and (not os.path.isfile(path)):
                    continue

                self.samples.append((path, int(wnid_to_idx[wnid])))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded. Check paths:\n"
                               f"train_root={train_root}\n"
                               f"csv={train_solution_csv}")

        # optional: sanity check
        labels = [y for _, y in self.samples[:5000]]  # 일부만
        print(f"[TrainFromLoc] #samples={len(self.samples)}, "
              f"label_min={min(labels)}, label_max={max(labels)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageNetValFromLocSolution(Dataset):
    """
    val 이미지: {val_root}/{ImageId}.JPEG   (ImageId = ILSVRC2012_val_000xxxxx)
    라벨: LOC_val_solution.csv의 PredictionString 첫 wnid를 synset mapping으로 0~999로 변환
    """
    def __init__(self, val_root, val_solution_csv, wnid_to_idx,
                 transform=None, ext=".JPEG", check_exists=True):
        self.val_root = val_root
        self.transform = transform
        self.ext = ext
        self.samples = []

        with open(val_solution_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["ImageId"]
                pred = row["PredictionString"].strip()
                if not pred:
                    continue

                wnid = pred.split()[0]
                if wnid not in wnid_to_idx:
                    continue

                path = os.path.join(val_root, image_id + ext)
                if check_exists and (not os.path.isfile(path)):
                    continue

                self.samples.append((path, int(wnid_to_idx[wnid])))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded. Check paths:\n"
                               f"val_root={val_root}\n"
                               f"csv={val_solution_csv}")

        # optional sanity check
        labels = [y for _, y in self.samples[:5000]]
        print(f"[ValFromLoc] #samples={len(self.samples)}, "
              f"label_min={min(labels)}, label_max={max(labels)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label



# def get_imagenet(batch_size,
#                  data_root='/ssdl/neurosim_data/ImageNet/extracted_data/ILSVRC',  # ILSVRC 루트로 받음
#                  loc_root='/ssdl/neurosim_data/ImageNet/extracted_data',
#                  val_ratio=0.1,
#                  **kwargs):

#     import os
#     from torchvision import transforms
#     from torch.utils.data import DataLoader, random_split

#     num_workers = kwargs.get('num_workers', 4)
#     pin_memory = kwargs.get('pin_memory', True)

#     # ✅ data_root가 ILSVRC 루트인지, CLS-LOC 루트인지 자동 판별
#     cand1 = os.path.join(data_root, "Data/CLS-LOC")   # ILSVRC 루트일 때
#     cand2 = data_root                                # 이미 CLS-LOC일 때

#     if os.path.isdir(os.path.join(cand1, "train")) and os.path.isdir(os.path.join(cand1, "val")):
#         cls_loc_root = cand1
#     elif os.path.isdir(os.path.join(cand2, "train")) and os.path.isdir(os.path.join(cand2, "val")):
#         cls_loc_root = cand2
#     else:
#         raise RuntimeError(f"Cannot find CLS-LOC train/val under: {data_root}")

#     train_root = os.path.join(cls_loc_root, "train")
#     val_root   = os.path.join(cls_loc_root, "val")

#     # transforms
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
#     test_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])

#     synset_path = os.path.join(loc_root, "LOC_synset_mapping.txt")
#     train_sol   = os.path.join(loc_root, "LOC_train_solution.csv")
#     val_sol     = os.path.join(loc_root, "LOC_val_solution.csv")

#     wnid_to_idx = build_wnid_to_idx(synset_path)

#     full_trainset = ImageNetTrainFromLocSolution(
#         train_root=train_root,
#         train_solution_csv=train_sol,
#         wnid_to_idx=wnid_to_idx,
#         transform=train_transform,
#         ext=".JPEG"
#     )

#     val_size = int(len(full_trainset) * val_ratio)
#     train_size = len(full_trainset) - val_size
#     train_set, val_set = random_split(full_trainset, [train_size, val_size])

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers, pin_memory=pin_memory)
#     val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers, pin_memory=pin_memory)

#     test_set = ImageNetValFromLocSolution(
#         val_root=val_root,
#         val_solution_csv=val_sol,
#         wnid_to_idx=wnid_to_idx,
#         transform=test_transform,
#         ext=".JPEG"
#     )
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
#                              num_workers=num_workers, pin_memory=pin_memory)

#     return train_loader, val_loader, test_loader

def get_imagenet(batch_size,
                 data_root='/ssdl/neurosim_data/ImageNet/extracted_data/ILSVRC',
                 loc_root='/ssdl/neurosim_data/ImageNet/extracted_data',
                 val_ratio=0.1,
                 seed=117,
                 **kwargs):

    num_workers = kwargs.get('num_workers', 4)
    pin_memory = kwargs.get('pin_memory', True)

    # # ✅ data_root 판별: "그 자체가 CLS-LOC인지" 먼저 확인 (중복 경로 방지)
    # cand_cls_loc = data_root
    # cand_ilsvrc  = os.path.join(data_root, "Data/CLS-LOC")

    # if os.path.isdir(os.path.join(cand_cls_loc, "train")) and os.path.isdir(os.path.join(cand_cls_loc, "val")):
    #     cls_loc_root = cand_cls_loc
    # elif os.path.isdir(os.path.join(cand_ilsvrc, "train")) and os.path.isdir(os.path.join(cand_ilsvrc, "val")):
    #     cls_loc_root = cand_ilsvrc
    # else:
    #     raise RuntimeError(f"Cannot find CLS-LOC train/val under: {data_root}")
    
    cls_loc_root = os.path.join(data_root, "Data/CLS-LOC")

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
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # LOC files
    synset_path = os.path.join(loc_root, "LOC_synset_mapping.txt")
    train_sol   = os.path.join(loc_root, "LOC_train_solution.csv")
    val_sol     = os.path.join(loc_root, "LOC_val_solution.csv")

    for p in [synset_path, train_sol, val_sol]:
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing file: {p}")

    wnid_to_idx = build_wnid_to_idx(synset_path)

    # ---- Train dataset (train_transform) ----
    full_trainset = ImageNetTrainFromLocSolution(
        train_root=train_root,
        train_solution_csv=train_sol,
        wnid_to_idx=wnid_to_idx,
        transform=train_transform,
        ext=".JPEG",
        check_exists=True
    )

    # ---- Split (reproducible) ----
    val_size = int(len(full_trainset) * val_ratio)
    train_size = len(full_trainset) - val_size
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_trainset, [train_size, val_size], generator=g)

    # ---- IMPORTANT: val_set은 eval_transform 사용 ----
    # random_split은 Subset이라 dataset은 동일 객체(full_trainset)를 공유함.
    # 그래서 val용으로 "별도 dataset"을 하나 더 만들어서 같은 csv를 읽되 transform만 eval로 둔다.
    full_trainset_eval = ImageNetTrainFromLocSolution(
        train_root=train_root,
        train_solution_csv=train_sol,
        wnid_to_idx=wnid_to_idx,
        transform=eval_transform,
        ext=".JPEG",
        check_exists=True
    )
    # 동일 split 인덱스를 그대로 사용
    val_set = torch.utils.data.Subset(full_trainset_eval, val_set.indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # ---- Test dataset (val_root + LOC_val_solution.csv) ----
    test_set = ImageNetValFromLocSolution(
        val_root=val_root,
        val_solution_csv=val_sol,
        wnid_to_idx=wnid_to_idx,
        transform=eval_transform,
        ext=".JPEG",
        check_exists=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # quick sanity check (optional but 추천)
    _, y = next(iter(train_loader))
    print("Train target min/max:", int(y.min()), int(y.max()))

    return train_loader, val_loader, test_loader
