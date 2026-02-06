"""Triplet dataset helper for metric learning.

Given an `image_list` and `label_list`, returns:
    (anchor_img, positive_img, negative_img, anchor_label)

Positive is sampled from the same label; negative from a different label.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, image_list: List[str], label_list: List[int], transform=None) -> None:
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

        self.label_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.label_list):
            self.label_to_indices.setdefault(label, []).append(idx)

        if len(self.label_to_indices) < 2:
            raise ValueError("TripletDataset requires at least 2 unique labels for negative sampling.")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int):
        anchor_path = self.image_list[index]
        anchor_label = self.label_list[index]

        # Positive: same label, different index
        pos_index = index
        candidates = self.label_to_indices[anchor_label]
        while pos_index == index:
            pos_index = random.choice(candidates)
        pos_path = self.image_list[pos_index]

        # Negative: different label
        neg_label = anchor_label
        all_labels = list(self.label_to_indices.keys())
        while neg_label == anchor_label:
            neg_label = random.choice(all_labels)
        neg_path = self.image_list[random.choice(self.label_to_indices[neg_label])]

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(pos_path).convert("RGB")
        negative = Image.open(neg_path).convert("RGB")

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, anchor_label
