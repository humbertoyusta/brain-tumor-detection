import prefect
import albumentations
from torch.utils.data import DataLoader
from typing import Tuple
import preprocessing.data_processor


@prefect.task
def process_data(
    batch_size: int = 16, num_workers: int = 0
) -> Tuple[
    DataLoader, DataLoader, DataLoader, albumentations.Compose, albumentations.Compose
]:
    processor = preprocessing.data_processor.DataProcessor(
        batch_size=batch_size, num_workers=num_workers
    )
    train_loader, val_loader, test_loader = processor.run()
    return (
        train_loader,
        val_loader,
        test_loader,
        processor.train_transform,
        processor.test_transform,
    )
