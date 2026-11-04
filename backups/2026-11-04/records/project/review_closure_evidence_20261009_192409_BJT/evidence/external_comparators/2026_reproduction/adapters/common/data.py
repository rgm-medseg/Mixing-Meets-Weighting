from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom


def read_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _augment(image: np.ndarray, label: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    if random.random() > 0.5:
        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        return np.flip(np.rot90(image, k), axis=axis).copy(), np.flip(np.rot90(label, k), axis=axis).copy()
    second = random.random()
    rotate = second <= 0.5 if variant == "hien" else second > 0.5
    if rotate:
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def _resize(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(array.shape) == shape:
        return array
    return zoom(array, (shape[0] / array.shape[0], shape[1] / array.shape[1]), order=0)


class FrozenTrainDataset:
    """Manifest-only train data surface; unlabeled entries never read a label."""

    def __init__(self, config: dict[str, Any], method: str):
        self.config = config
        self.method = method.lower()
        self.dataset = config["dataset"]
        self.data = config["data"]
        self.patch_size = tuple(int(v) for v in config["patch_size"])
        self.entries = self._entries(labeled=True) + self._entries(labeled=False)
        self.labeled_count = sum(bool(entry["is_labeled"]) for entry in self.entries)
        self.unlabeled_count = len(self.entries) - self.labeled_count
        self.labeled_indices = list(range(self.labeled_count))
        self.unlabeled_indices = list(range(self.labeled_count, len(self.entries)))
        self._promise_images: np.ndarray | None = None
        self._promise_labels: np.ndarray | None = None

    def _entries(self, labeled: bool) -> list[dict[str, Any]]:
        path = self.data["labeled_manifest" if labeled else "unlabeled_manifest"]
        if self.dataset == "ACDC":
            return [{"sample_id": item, "is_labeled": labeled} for item in read_lines(path)]
        rows = read_tsv(path)
        return [{**row, "is_labeled": labeled} for row in rows]

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        labeled = bool(entry["is_labeled"])
        if self.dataset == "ACDC":
            path = Path(self.data["root"]) / "data" / "slices" / f"{entry['sample_id']}.h5"
            with h5py.File(path, "r") as handle:
                image = handle["image"][:].astype(np.float32)
                label = handle["label"][:].astype(np.uint8) if labeled else np.zeros(image.shape, dtype=np.uint8)
            return image, label
        if self.dataset == "PROMISE12":
            if self._promise_images is None:
                self._promise_images = np.load(self.data["train_images"], mmap_mode="r")
            index = int(entry["array_index"])
            image = np.asarray(self._promise_images[index], dtype=np.float32)
            if labeled:
                if self._promise_labels is None:
                    self._promise_labels = np.load(self.data["train_labels"], mmap_mode="r")
                label = np.asarray(self._promise_labels[index], dtype=np.uint8)
            else:
                label = np.zeros(image.shape, dtype=np.uint8)
            return image, label
        path = Path(entry["derived_path"])
        if not path.is_absolute():
            path = Path(self.data["project_root"]) / path
        payload = np.load(path)
        image = payload["image"].astype(np.float32)
        label = payload["mask"].astype(np.uint8) if labeled else np.zeros(image.shape, dtype=np.uint8)
        return image, label

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image, label = self._load(entry)
        image, label = _augment(image, label, self.method)
        image = _resize(image, self.patch_size).astype(np.float32)
        label = _resize(label, self.patch_size).astype(np.uint8)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "label": torch.from_numpy(label).long(),
            "is_labeled": bool(entry["is_labeled"]),
            "sample_id": entry.get("sample_id", entry.get("slice_id", str(index))),
            "patient_id": entry.get("patient_id", entry.get("case_id", "")),
        }

    def batch(self, indices: list[int]) -> dict[str, Any]:
        samples = [self[index] for index in indices]
        return {
            "image": torch.stack([sample["image"] for sample in samples]),
            "label": torch.stack([sample["label"] for sample in samples]),
            "is_labeled": torch.tensor([sample["is_labeled"] for sample in samples], dtype=torch.bool),
            "sample_id": [sample["sample_id"] for sample in samples],
            "patient_id": [sample["patient_id"] for sample in samples],
        }


class StatefulTwoStream:
    """Two shuffled streams with a serializable exact next-batch state."""

    def __init__(self, labeled: list[int], unlabeled: list[int], labeled_bs: int, unlabeled_bs: int, seed: int):
        if min(len(labeled), len(unlabeled), labeled_bs, unlabeled_bs) <= 0:
            raise ValueError("both two-stream partitions and batch portions must be non-empty")
        self.sources = [np.asarray(labeled), np.asarray(unlabeled)]
        self.batch_sizes = [int(labeled_bs), int(unlabeled_bs)]
        self.rng = np.random.default_rng(seed)
        self.permutations = [self.rng.permutation(source) for source in self.sources]
        self.positions = [0, 0]
        self.batches_emitted = 0

    def _take(self, stream: int) -> list[int]:
        result: list[int] = []
        needed = self.batch_sizes[stream]
        while len(result) < needed:
            remaining = len(self.permutations[stream]) - self.positions[stream]
            if remaining == 0:
                self.permutations[stream] = self.rng.permutation(self.sources[stream])
                self.positions[stream] = 0
                remaining = len(self.permutations[stream])
            take = min(needed - len(result), remaining)
            start = self.positions[stream]
            result.extend(int(value) for value in self.permutations[stream][start:start + take])
            self.positions[stream] += take
        return result

    def next_batch(self) -> list[int]:
        self.batches_emitted += 1
        return self._take(0) + self._take(1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.bit_generator.state,
            "permutations": [value.tolist() for value in self.permutations],
            "positions": list(self.positions),
            "batches_emitted": self.batches_emitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.permutations = [np.asarray(value, dtype=np.int64) for value in state["permutations"]]
        self.positions = [int(value) for value in state["positions"]]
        self.batches_emitted = int(state["batches_emitted"])


def iter_volumes(config: dict[str, Any], split: str) -> Iterator[dict[str, Any]]:
    data = config["data"]
    dataset = config["dataset"]
    manifest = data[f"{split}_manifest"]
    if dataset == "ACDC":
        for case_id in read_lines(manifest):
            path = Path(data["root"]) / "data" / f"{case_id}.h5"
            with h5py.File(path, "r") as handle:
                image = handle["image"][:].astype(np.float32)
                label = handle["label"][:].astype(np.uint8)
                spacing = handle.attrs.get("spacing")
            yield {"case_id": case_id, "image": image, "label": label, "spacing": spacing}
        return
    if dataset == "PROMISE12":
        root = Path(data["root"]) / "npy_image"
        for case_id in read_lines(manifest):
            yield {
                "case_id": case_id,
                "image": np.load(root / f"{case_id}.npy").astype(np.float32),
                "label": np.load(root / f"{case_id}_segmentation.npy").astype(np.uint8),
                "spacing": None,
            }
        return
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_tsv(manifest):
        grouped.setdefault(row["patient_id"], []).append(row)
    for patient_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: int(item["slice_idx"]))
        images, labels, spacings = [], [], []
        for row in rows:
            path = Path(row["derived_path"])
            if not path.is_absolute():
                path = Path(data["project_root"]) / path
            payload = np.load(path)
            images.append(payload["image"].astype(np.float32))
            labels.append(payload["mask"].astype(np.uint8))
            if "spacing" in payload.files:
                spacings.append(tuple(float(value) for value in payload["spacing"].tolist()))
            else:
                spacings.append(tuple(float(value) for value in row["canonical_spacing"].split(",")))
        if len(set(spacings)) != 1:
            raise ValueError(f"inconsistent spacing for {patient_id}")
        yield {
            "case_id": patient_id,
            "image": np.stack(images),
            "label": np.stack(labels),
            "spacing": spacings[0],
        }
