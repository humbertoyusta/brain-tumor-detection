import cv2
import albumentations


class CropBrainRegion(albumentations.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, output_size, always_apply=True, p=1.0, return_everything=False):
        """
        Custom transform to crop the brain region from the MRI scan image.
        Args:
            output_size (tuple): Desired output size. It should be a tuple like (height, width).
            always_apply (bool): Whether to always apply the transform.
            p (float): Probability of applying the transform.
            return_everything (bool): Whether to return the cropped image along with the bounding box and contours.
        Returns:
            Cropped image if return_everything is False, otherwise a tuple of (cropped_image, contour_image, bounding_box_image, original_cropped_image).
        """
        super(CropBrainRegion, self).__init__(always_apply, p)
        self.size = output_size
        self.return_everything = return_everything

    def apply(self, image, **params):
        # Converting the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Applying Gaussian blur to smooth the image and reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding the image to create a binary mask
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

        # Performing morphological operations to remove noise
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Finding contours in the binary mask
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Assuming the brain part of the image has the largest contour
        c = max(contours, key=cv2.contourArea)

        # Getting the bounding rectangle of the brain part
        x, y, w, h = cv2.boundingRect(c)

        if self.return_everything:
            # Drawing contours on the original image
            contour_image = cv2.drawContours(image.copy(), [c], -1, (0, 255, 0), 2)

            # Drawing bounding box on the original image
            bounding_box_image = cv2.rectangle(
                image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2
            )

        # Cropping the image around the bounding rectangle
        cropped_image = image[y : y + h, x : x + w]

        # Resizing cropped image to the needed size
        resized_image = cv2.resize(cropped_image, self.size)

        if self.return_everything:
            return resized_image, contour_image, bounding_box_image, cropped_image
        else:
            return resized_image

    def get_transform_init_args_names(self):
        return ("size",)
