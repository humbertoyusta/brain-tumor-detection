import os
import torch
import albumentations
import albumentations.pytorch
import cv2
import preprocessing.constants
import preprocessing.crop_brain_region


class ModelExecutor:
    def __init__(self):
        self.device = self._get_device()
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_transform(self):
        return albumentations.Compose(
            [
                preprocessing.crop_brain_region.CropBrainRegion(
                    output_size=preprocessing.constants.IMAGE_SIZE
                ),
                albumentations.Normalize(
                    mean=preprocessing.constants.MEAN, std=preprocessing.constants.STD
                ),
                albumentations.pytorch.ToTensorV2(),
            ]
        )

    def _load_model(self):
        model = torch.load(os.path.join("app", "model.pth"))
        model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image):
        image = cv2.resize(image, preprocessing.constants.IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        image = image.unsqueeze(0)
        return image.to(self.device)

    def predict(self, image):
        image = self._preprocess_image(image)
        with torch.no_grad():
            logit = self.model(image)
            return {
                "logit": logit.cpu().numpy().tolist()[0][0],
                "prediction": (logit > 0.5).cpu().numpy().tolist()[0][0],
            }
