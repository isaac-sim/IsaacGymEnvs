import math
from typing import Tuple, List, Optional
from torch import Tensor

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms._functional_tensor import _max_value, rgb_to_grayscale, _assert_image_tensor, \
    _assert_channels, get_dimensions, convert_image_dtype, _rgb2hsv, _hsv2rgb


class ColorJitterStateful(T.ColorJitter):

    def __init__(self, num_envs, device, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.num_envs = num_envs
        self.device = device
        self.fn_idx, self.brightness_factor, self.contrast_factor, self.saturation_factor, self.hue_factor = [None] * 5
        self.sample_transform()

    def sample_transform(self):
        self.fn_idx, self.brightness_factor, self.contrast_factor, self.saturation_factor, self.hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

    def get_params(self,
            brightness: Optional[List[float]],
            contrast: Optional[List[float]],
            saturation: Optional[List[float]],
            hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else torch.empty(self.num_envs, device=self.device).uniform_(brightness[0], brightness[1])
        c = None if contrast is None else torch.empty(self.num_envs, device=self.device).uniform_(contrast[0], contrast[1])
        s = None if saturation is None else torch.empty(self.num_envs, device=self.device).uniform_(saturation[0], saturation[1])
        h = None if hue is None else torch.empty(self.num_envs, device=self.device).uniform_(hue[0], hue[1])

        return fn_idx, b, c, s, h

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness_factor is not None:
                img = adjust_brightness_batch(img, self.brightness_factor)
            elif fn_id == 1 and self.contrast_factor is not None:
                img = adjust_contrast_batch(img, self.contrast_factor)
            elif fn_id == 2 and self.saturation_factor is not None:
                img = adjust_saturation_batch(img, self.saturation_factor)
            elif fn_id == 3 and self.hue_factor is not None:
                img = adjust_hue(img, self.hue_factor)

        return img


def adjust_brightness_batch(image: Tensor, brightness_factor: Tensor) -> Tensor:
    if any(brightness_factor < 0):
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")

    _assert_image_tensor(image)

    _assert_channels(image, [1, 3])

    return _blend_batch(image, torch.zeros_like(image), brightness_factor)


def adjust_contrast_batch(img: Tensor, contrast_factor: Tensor) -> Tensor:
    if any(contrast_factor < 0):
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")

    _assert_image_tensor(img)

    _assert_channels(img, [3, 1])
    c = get_dimensions(img)[0]
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    if c == 3:
        mean = torch.mean(rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)
    else:
        mean = torch.mean(img.to(dtype), dim=(-3, -2, -1), keepdim=True)

    return _blend_batch(img, mean, contrast_factor)


def adjust_saturation_batch(img: Tensor, saturation_factor: Tensor) -> Tensor:
    if any(saturation_factor < 0):
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])

    if get_dimensions(img)[0] == 1:  # Match PIL behaviour
        return img

    return _blend_batch(img, rgb_to_grayscale(img), saturation_factor)


def adjust_hue(img: Tensor, hue_factor: Tensor) -> Tensor:
    if any(torch.logical_and(hue_factor < -0.5, hue_factor > 0.5)):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    if not (isinstance(img, Tensor)):
        raise TypeError("Input img should be Tensor image")

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])
    if get_dimensions(img)[0] == 1:  # Match PIL behaviour
        return img

    orig_dtype = img.dtype
    img = convert_image_dtype(img, torch.float32)

    img = _rgb2hsv(img)
    h, s, v = img.unbind(dim=-3)
    h = (h + hue_factor[:, None, None]) % 1.0
    img = torch.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv2rgb(img)

    return convert_image_dtype(img_hue_adj, orig_dtype)


def _blend_batch(img1: Tensor, img2: Tensor, ratio: Tensor) -> Tensor:
    bound = _max_value(img1.dtype)
    ratio_batch = ratio[:, None, None, None]
    return (ratio_batch * img1 + (1.0 - ratio_batch) * img2).clamp(0, bound).to(img1.dtype)


class RandomResizedCropStateful(T.RandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=F.InterpolationMode.BILINEAR):
        super().__init__(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.i, self.j, self.h, self.w = [None] * 4

    def sample_transform(self, img_width, img_height):
        self.i, self.j, self.h, self.w = self.get_params(img_width, img_height, self.scale, self.ratio)

    @staticmethod
    def get_params(
            img_height: int, img_width: int, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img_width (int): Height of input image.
            img_height (int): Height of input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        height, width = img_height, img_width
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        return F.resized_crop(img, self.i, self.j, self.h, self.w, self.size, self.interpolation)
