import torch
import prefect
import utils.device


@prefect.task
def get_device() -> torch.device:
    return utils.device.get_device()
