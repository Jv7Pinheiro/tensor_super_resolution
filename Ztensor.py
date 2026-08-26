from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np


@dataclass
class Dataset:
    Z: np.ndarray
    N: int
    T_max: float
    shots: int
    sigma: float
    alpha: float
    dx: float
    tau: float
    t_list: np.ndarray

    def __post_init__(self) -> None:
        self.t_list = np.asarray(self.t_list)

        if self.N != len(self.t_list):
            raise ValueError(
                f"N ({self.N}) must match len(t_list) ({len(self.t_list)})"
            )

        if self.Z.shape[-1] != self.N:
            raise ValueError(
                f"The last dimension of Z ({self.Z.shape[-1]}) "
                f"must match N ({self.N})"
            )

        if self.shots < 1:
            raise ValueError("shots must be at least 1")

        if self.T_max <= 0:
            raise ValueError("T_max must be positive")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.Z.shape

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        with Path(path).open("rb") as file:
            dataset = pickle.load(file)

        if not isinstance(dataset, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")

        return dataset

    def __repr__(self) -> str:
        return (
            f"Dataset(shape={self.Z.shape}, N={self.N}, "
            f"T_max={self.T_max}, shots={self.shots})"
        )