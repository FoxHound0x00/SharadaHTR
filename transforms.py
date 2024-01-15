from skimage import transform, color, filters
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import Normalize
import torchvision.transforms.functional as F

class PadResize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        self.w_f, self.h_f = self.output_size
        ratio_final = self.w_f / self.h_f
        self.w, self.h = image.size
        self.ratio_current = self.w / self.h

        # check if the original and final aspect ratios are the same within a margin
        if round(self.ratio_current, 2) != round(ratio_final, 2):
            # padding to preserve aspect ratio
            hp = int(self.w/ratio_final - self.h)
            wp = int(ratio_final * self.h - self.w)

            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                image = F.resize(image, [self.h_f, self.w_f])
            elif wp > 0 and hp < 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                image = F.resize(image, [self.h_f, self.w_f])
        else:
            image = F.resize(image,[self.h_f, self.w_f])

        return {'image': image, 'label': label}

class Deskew(object):
    """Deskew handwriting samples"""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        try:
            threshold = filters.threshold_otsu(image)
        except ValueError:
            return {'image':image, 'label':label}

        binary = image.copy() < threshold

        # array of alpha values
        alphas = np.arange(-1, 1.1, 0.25)
        alpha_res = np.array([])
        alpha_params = []

        for a in alphas:
            alpha_sum = 0
            shift_x = np.max([-a*binary.shape[0], 0])
            M = np.array([[1, a, shift_x],
                          [0,1,0]], dtype=np.float64)
            img_size = (np.int(binary.shape[1] + np.ceil(np.abs(a*binary.shape[0]))), binary.shape[0])
            alpha_params.append((M, img_size))


            img_shear = cv.warpAffine(src=binary.astype(np.uint8),
                                      M=M, dsize=img_size,
                                      flags=cv.INTER_NEAREST)

            for i in range(0, img_shear.shape[1]):
                if not np.any(img_shear[:, i]):
                    continue

                h_alpha = np.sum(img_shear[:, i])
                fgr_pos = np.where(img_shear[:, i] == 1)
                delta_y_alpha = fgr_pos[0][-1] - fgr_pos[0][0] + 1

                if h_alpha == delta_y_alpha:
                    alpha_sum += h_alpha**2

            alpha_res = np.append(alpha_res, alpha_sum)

        best_M, best_size = alpha_params[alpha_res.argmax()]
        deskewed_img = cv.warpAffine(src=image, M=best_M, dsize=best_size,
                                      flags=cv.INTER_LINEAR,
                                      borderMode=cv.BORDER_CONSTANT,
                                      borderValue=255)

        return {'image':deskewed_img, 'label':label}

class toRGB(object):
    """Convert the ndarrys to RGB tensors.
       Required if using ImageNet pretrained Resnet."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = color.gray2rgb(image)

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, rgb=True):
        assert isinstance(rgb, bool)
        self.rgb = rgb

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = F.to_tensor(image)
        return {'image': image, 'label': label}



class Normalize_Cust(object):
    """Normalise by channel mean and std"""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)
        self.norm = Normalize(mean, std)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': self.norm(image), 'label': label}