from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset, DataLoader
from torchvision import datasets, transforms

@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    classes: Tuple[int, ...]

CIFAR10_TASKS = [
    TaskSpec(0, "c10_01", (0,1)),
    TaskSpec(1, "c10_23", (2,3)),
    TaskSpec(2, "c10_45", (4,5)),
    TaskSpec(3, "c10_67", (6,7)),
    TaskSpec(4, "c10_89", (8,9)),
]

# Standard CIFAR-100 coarse groups, expressed by fine-class names.
CIFAR100_COARSE = {
"aquatic_mammals": ["beaver","dolphin","otter","seal","whale"],
"fish": ["aquarium_fish","flatfish","ray","shark","trout"],
"flowers": ["orchid","poppy","rose","sunflower","tulip"],
"food_containers": ["bottle","bowl","can","cup","plate"],
"fruit_and_vegetables": ["apple","mushroom","orange","pear","sweet_pepper"],
"household_electrical_devices": ["clock","keyboard","lamp","telephone","television"],
"household_furniture": ["bed","chair","couch","table","wardrobe"],
"insects": ["bee","beetle","butterfly","caterpillar","cockroach"],
"large_carnivores": ["bear","leopard","lion","tiger","wolf"],
"large_man_made_outdoor_things": ["bridge","castle","house","road","skyscraper"],
"large_natural_outdoor_scenes": ["cloud","forest","mountain","plain","sea"],
"large_omnivores_and_herbivores": ["camel","cattle","chimpanzee","elephant","kangaroo"],
"medium_sized_mammals": ["fox","porcupine","possum","raccoon","skunk"],
"non_insect_invertebrates": ["crab","lobster","snail","spider","worm"],
"people": ["baby","boy","girl","man","woman"],
"reptiles": ["crocodile","dinosaur","lizard","snake","turtle"],
"small_mammals": ["hamster","mouse","rabbit","shrew","squirrel"],
"trees": ["maple_tree","oak_tree","palm_tree","pine_tree","willow_tree"],
"vehicles_1": ["bicycle","bus","motorcycle","pickup_truck","train"],
"vehicles_2": ["lawn_mower","rocket","streetcar","tank","tractor"],
}

# Ten semantically structured tasks, each pairing two related CIFAR-100 superclasses.
CIFAR100_SEMANTIC_PAIRS = [
    ("aquatic_mammals", "fish"),
    ("flowers", "fruit_and_vegetables"),
    ("food_containers", "household_electrical_devices"),
    ("household_furniture", "large_man_made_outdoor_things"),
    ("insects", "non_insect_invertebrates"),
    ("large_carnivores", "large_omnivores_and_herbivores"),
    ("large_natural_outdoor_scenes", "trees"),
    ("medium_sized_mammals", "small_mammals"),
    ("people", "reptiles"),
    ("vehicles_1", "vehicles_2"),
]


def _tfm(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    return transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])


def load_dataset(name: str, root: str, train: bool, download: bool=True, augment: bool|None=None) -> Dataset:
    """Load CIFAR with explicit augmentation control.

    For training data, augment=True uses random crop/flip while augment=False
    uses the deterministic evaluation transform. This is used to keep resident
    matching and pre-invasion probes free of augmentation noise.
    """
    name = name.lower()
    use_train_tfm = train if augment is None else bool(augment)
    tfm = _tfm(use_train_tfm)
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, transform=tfm, download=download)
    if name == "cifar100":
        return datasets.CIFAR100(root, train=train, transform=tfm, download=download)
    raise ValueError(name)


def semantic_cifar100_tasks(ds: datasets.CIFAR100) -> List[TaskSpec]:
    name_to_id = {n:i for i,n in enumerate(ds.classes)}
    tasks=[]
    for tid,(a,b) in enumerate(CIFAR100_SEMANTIC_PAIRS):
        names = CIFAR100_COARSE[a] + CIFAR100_COARSE[b]
        missing = [n for n in names if n not in name_to_id]
        if missing:
            raise KeyError(f"CIFAR100 class names not found: {missing}; available sample={ds.classes[:10]}")
        tasks.append(TaskSpec(tid, f"c100_{a}+{b}", tuple(name_to_id[n] for n in names)))
    return tasks


def task_subset(ds: Dataset, task: TaskSpec) -> Subset:
    targets = np.asarray(getattr(ds, "targets"))
    idx = np.flatnonzero(np.isin(targets, np.asarray(task.classes))).tolist()
    return Subset(ds, idx)


def class_balanced_indices(ds: Dataset, classes: Sequence[int], n_total: int, seed: int) -> List[int]:
    targets = np.asarray(getattr(ds, "targets"))
    rng = np.random.default_rng(seed)
    per = n_total // len(classes)
    rem = n_total - per*len(classes)
    out=[]
    for j,c in enumerate(classes):
        ids = np.flatnonzero(targets == c)
        k = min(len(ids), per + (1 if j < rem else 0))
        out.extend(rng.choice(ids, size=k, replace=False).tolist())
    rng.shuffle(out)
    return out


def loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int=2, seed: int=0) -> DataLoader:
    g=torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),generator=g)


def synthetic_tasks(seed:int=0, n_per_class:int=80, n_classes:int=6):
    """Offline smoke dataset with learnable class structure."""
    g=torch.Generator().manual_seed(seed)
    xs=[]; ys=[]
    for c in range(n_classes):
        base=torch.zeros(3,32,32)
        base[:, (c*4)%24:(c*4)%24+8, (c*5)%24:(c*5)%24+8] = 1.0
        x=base.unsqueeze(0)+0.20*torch.randn(n_per_class,3,32,32,generator=g)
        xs.append(x); ys.append(torch.full((n_per_class,),c,dtype=torch.long))
    X=torch.cat(xs); Y=torch.cat(ys)
    ds=TensorDataset(X,Y)
    tasks=[TaskSpec(i,f"syn_{2*i}_{2*i+1}",(2*i,2*i+1)) for i in range(n_classes//2)]
    def subset(task):
        m=torch.zeros_like(Y,dtype=torch.bool)
        for c in task.classes: m |= (Y==c)
        return Subset(ds, torch.nonzero(m,as_tuple=False).flatten().tolist())
    return ds,tasks,subset
