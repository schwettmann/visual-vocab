"""Datasets for reading directions and annotations from the paper."""
import collections
import csv
import pathlib
import tempfile
from typing import Any, Mapping, NamedTuple, Optional

from visualvocab.utils import env
from visualvocab.utils.typing import Device, PathLike, StrSequence

import torch
from torch.utils import data
from torchvision.datasets import utils


class Direction(NamedTuple):
    """A solitary GAN direction."""

    id: int
    direction: torch.Tensor


class DirectionsDataset(data.Dataset[Direction]):
    """A dataset of GAN directions."""

    def __init__(self,
                 root: PathLike,
                 ds_file_name: str = 'ds.pth',
                 metadata_file_name: str = 'metadata.csv',
                 device: Device = 'cpu'):
        """Initialize the dataset.

        Args:
            root (PathLike): Root directory containing all files.
            ds_file_name (str, optional): Name of .pth file containing
                directions. Defaults to 'ds.pth'.
            metadata_file_name (str, optional): Name of .csv file containing
                directions metadata. Defaults to 'metadata.csv'.
            device (Device, optional): Send all tensors to this device.

        """
        self.root = root = pathlib.Path(root)

        # Validate the necessary files all exist.
        ds_file = root / ds_file_name
        metadata_file = root / metadata_file_name
        for key, file in (
            ('ds', ds_file),
            ('metadata', metadata_file),
        ):
            if not file.exists():
                raise FileNotFoundError(f'{key} file not found: {file}')

        # Load zs and ds.
        self.ds = torch.load(ds_file, map_location=device)
        if not isinstance(self.ds, torch.Tensor):
            raise ValueError('directions file contains non-tensor')
        if self.ds.dim() != 2:
            raise ValueError(f'need 2D ds, got {self.ds.dim()}D')

        # Pre-wrap all directions for consistency.
        directions = []
        for d_id, d in enumerate(self.ds):
            direction = Direction(d_id, d)
            directions.append(direction)
        self.directions = tuple(directions)

        # Load and process metadata.
        with metadata_file.open('r') as handle:
            metadata_rows = tuple(csv.DictReader(handle))

        metadata_by_d_id = {}
        for row in metadata_rows:
            d_id = int(row['d_id'])
            metadata_by_d_id[d_id] = row

        d_ids = metadata_by_d_id.keys()
        missing = set(range(len(self.ds))) - d_ids
        if missing:
            raise ValueError(f'metadata csv missing directions: {missing}')

        self.metadata = [metadata_by_d_id[d_id] for d_id in sorted(d_ids)]

    def __getitem__(self, index: int) -> Direction:
        """Get the index-th direction in the dataset.

        Args:
            index (int): Direction index.

        Returns:
            Direction: The direction.

        """
        return Direction(index, self.ds[index])

    def __len__(self) -> int:
        """Return the number of directions in the dataset."""
        return len(self.ds)

    @property
    def dim(self) -> int:
        """Return the dimensionality of the directions."""
        return self.ds.shape[-1]


class AnnotatedDirection(NamedTuple):
    """A solitary GAN direction that is annotated with a language string."""

    id: int
    direction: torch.Tensor
    annotations: StrSequence


class AnnotatedDirectionsDataset(data.Dataset[AnnotatedDirection]):
    """A dataset of annotated GAN directions."""

    def __init__(self,
                 root: PathLike,
                 annotations_file_name: str = 'annotations.csv',
                 **kwargs: Any):
        """Initialize the dataset.

        The **kwargs are forwarded to DirectionsDataset.

        Args:
            root (PathLike): Root directory containing all files.
            annotations_file_name (str, optional): Name of .pth file containing
                directions. Defaults to 'ds.pth'.

        Raises:
            ValueError: If any direction is missing an annotation.

        """
        self.root = root = pathlib.Path(root)

        dataset = DirectionsDataset(root, **kwargs)

        annotations_file = root / annotations_file_name
        if not annotations_file.exists():
            raise FileNotFoundError(
                f'annotations file not found: {annotations_file}')
        with annotations_file.open('r') as handle:
            annotations_rows = tuple(csv.DictReader(handle))

        annotations_by_d_id = collections.defaultdict(list)
        for row in annotations_rows:
            d_id = int(row['d_id'])
            annotations_by_d_id[d_id].append(row['annotation'])

        directions = []
        for index in range(len(dataset)):
            annotations = tuple(annotations_by_d_id[index])
            direction = AnnotatedDirection(index, dataset.ds[index],
                                           annotations)
            directions.append(direction)
        self.directions = tuple(directions)

        self.dim = dataset.dim

    def __getitem__(self, index: int) -> AnnotatedDirection:
        """Get the index-th direction in the dataset.

        Args:
            index (int): The index.

        Returns:
            AnnotatedDirection: The direction.

        """
        return self.directions[index]

    def __len__(self) -> int:
        """Return the number of directions in the dataset."""
        return len(self.directions)


DATASETS_BASE_URL = 'https://visualvocab.csail.mit.edu'
DATASETS_URLS = {
    # The big dataset of the 1280 annotated LSDs that were used to distil
    # single-concept directions. Each direction appears 4 times in the dataset
    # because each was visualized and annotated once per category.
    'lsd_all': f'{DATASETS_BASE_URL}/lsd_all.zip',

    # Same as above, but only including annotations for specific directories,
    # if that's your thing.
    'lsd_cottage': f'{DATASETS_BASE_URL}/lsd_cottage.zip',
    'lsd_kitchen': f'{DATASETS_BASE_URL}/lsd_kitchen.zip',
    'lsd_lake': f'{DATASETS_BASE_URL}/lsd_lake.zip',
    'lsd_medina': f'{DATASETS_BASE_URL}/lsd_medina.zip',

    # Directions distilled from the lsd_2x dataset, using ALL the LSDs (_all)
    # or using only LSDs that were visualized in a specific category
    # (e.g., _cottage).
    'distilled_all': f'{DATASETS_BASE_URL}/distilled_all.zip',
    'distilled_cottage': f'{DATASETS_BASE_URL}/distilled_cottage.zip',
    'distilled_kitchen': f'{DATASETS_BASE_URL}/distilled_kitchen.zip',
    'distilled_lake': f'{DATASETS_BASE_URL}/distilled_lake.zip',
    'distilled_medina': f'{DATASETS_BASE_URL}/distilled_medina.zip',
}


def load(
    key: str,
    urls: Mapping[str, str] = DATASETS_URLS,
    data_dir: Optional[PathLike] = None,
    download: bool = True,
) -> AnnotatedDirectionsDataset:
    """Download directions annotated in the original paper.

    Args:
        key (str): The dataset key. See DATASETS_URLS for options.
        urls (Mapping[str, str], optional): Mapping from dataset key to the
            dataset URL. Defaults to DATASETS_URLS constant.

    Returns:
        AnnotatedDirectionsDataset: The loaded dataset.

    """
    url = urls.get(key)
    if url is None:
        raise KeyError(f'no such dataset "{key}"; '
                       f'options are {", ".join(sorted(urls.keys()))}')

    if data_dir is None:
        data_dir = env.data_dir() / key
    else:
        data_dir = pathlib.Path(data_dir)

    if not data_dir.exists() and download:
        data_dir.mkdir(exist_ok=True, parents=True)
        with tempfile.TemporaryDirectory() as tempdir:
            utils.download_and_extract_archive(url,
                                               pathlib.Path(tempdir),
                                               extract_root=data_dir)

    return AnnotatedDirectionsDataset(data_dir)
