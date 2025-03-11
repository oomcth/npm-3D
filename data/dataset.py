import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import glob


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class TabularDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        feature_cols: List[str],
        target_cols: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data = pd.read_csv(data_path)
        self.features = self.data[feature_cols].values
        self.targets = self.data[target_cols].values
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y


class ImageDataset(BaseDataset):
    def __init__(
        self,
        image_dir: str,
        annotations_file: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.image_dir = image_dir
        self.transform = transform

        if annotations_file:
            self.annotations = pd.read_csv(annotations_file)
            self.image_paths = [os.path.join(image_dir, filename) for filename in self.annotations["filename"]]
            self.labels = self.annotations["label"].values
        else:
            self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
            self.labels = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label

        return image

    def _load_image(self, path):
        # Implement image loading based on your needs (PIL, OpenCV, etc.)
        # For example, using PIL:
        from PIL import Image
        return Image.open(path).convert("RGB")


class LidarPointCloudDataset(BaseDataset):
    def __init__(self, root_dir, transform=None, max_points=None, cache_data=False):
        self.root_dir = root_dir
        self.transform = transform
        self.max_points = max_points
        self.cache_data = cache_data

        self.point_cloud_files = []
        for ext in ['.bin', '.ply', '.pcd', '.xyz', '.pts']:
            self.point_cloud_files.extend(glob.glob(os.path.join(root_dir, f'*{ext}')))
        self.point_cloud_files.sort()

        self.data_cache = {} if cache_data else None

    def __len__(self):
        return len(self.point_cloud_files)

    def __getitem__(self, idx):
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]

        file_path = self.point_cloud_files[idx]
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.bin':
            point_cloud = self._load_bin_file(file_path)
        elif file_extension == '.ply':
            point_cloud = self._load_ply_file(file_path)
        elif file_extension == '.pcd':
            point_cloud = self._load_pcd_file(file_path)
        elif file_extension in ['.xyz', '.pts']:
            point_cloud = self._load_txt_file(file_path)
        else:
            raise ValueError(f"Format de fichier non pris en charge: {file_extension}")

        if self.max_points is not None and point_cloud.shape[0] > self.max_points:
            indices = np.random.choice(point_cloud.shape[0], self.max_points, replace=False)
            point_cloud = point_cloud[indices]

        point_cloud_tensor = torch.from_numpy(point_cloud).float()

        if self.transform:
            point_cloud_tensor = self.transform(point_cloud_tensor)

        result = {
            'points': point_cloud_tensor,
            'file_path': file_path,
            'index': idx
        }

        if self.cache_data:
            self.data_cache[idx] = result

        return result

    def _load_bin_file(self, file_path):
        """Charge un fichier binaire contenant un nuage de points."""
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # [x, y, z, intensity]
        return points

    def _load_ply_file(self, file_path):
        """Charge un fichier PLY contenant un nuage de points."""
        try:
            from plyfile import PlyData
            plydata = PlyData.read(file_path)
            data = plydata['vertex'].data
            x = data['x']
            y = data['y']
            z = data['z']

            if 'intensity' in data.dtype.names:
                intensity = data['intensity']
                points = np.vstack((x, y, z, intensity)).T
            else:
                points = np.vstack((x, y, z)).T

            return points
        except ImportError:
            print("La bibliothèque 'plyfile' est requise pour charger les fichiers PLY. Installez-la avec pip install plyfile.")
            raise

    def _load_pcd_file(self, file_path):
        """Charge un fichier PCD contenant un nuage de points."""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)

            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                points = np.hstack((points, colors))

            return points
        except ImportError:
            print("La bibliothèque 'open3d' est requise pour charger les fichiers PCD. Installez-la avec pip install open3d.")
            raise

    def _load_txt_file(self, file_path):
        """Charge un fichier texte (xyz ou pts) contenant un nuage de points."""
        points = np.loadtxt(file_path, delimiter=' ')
        return points

    def get_all_points(self):
        all_points = []
        cloud_ids = []

        for i in range(len(self)):
            data = self[i]
            points = data['points'].numpy()
            all_points.append(points)
            cloud_ids.append(np.full(len(points), i))

        return np.vstack(all_points), np.concatenate(cloud_ids)


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader
