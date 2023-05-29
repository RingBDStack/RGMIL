import numpy as np
import numpy.random as random
from PIL import Image, ImageFilter
from skimage import color
import skimage

class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomRotate(object):
    def __call__(self, img):
        random_rotation = random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = img.rotate(random_rotation*90)
        return img

class RandomHEStain(object):
    def __call__(self, img):
        img_he = skimage.color.rgb2hed(img)
        img_he[:, :, 0] = img_he[:, :, 0] * random.normal(1.0, 0.02, 1)
        img_he[:, :, 1] = img_he[:, :, 1] * random.normal(1.0, 0.02, 1)
        img_rgb = np.clip(skimage.color.hed2rgb(img_he), 0, 1)
        img = Image.fromarray(np.uint8(img_rgb*255.999))
        return img

class RandomGaussianNoise(object):
    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(random.normal(0.0, 0.5, 1)))
        return img

class HistoNormalize(object):
    def __call__(self, img):
        img_arr = np.array(img)
        img_norm = normalize(img_arr)
        img = Image.fromarray(img_norm)
        return img

def normalize(image, target=None):
    if target is None:
        target = np.array([[57.4, 15.84], [39.9, 9.14], [-22.34, 6.58]])
    whitemask = color.rgb2gray(image)
    whitemask = whitemask > (215 / 255)
    imagelab = color.rgb2lab(image)
    imageL, imageA, imageB = [imagelab[:, :, i] for i in range(3)]
    imageLM = np.ma.MaskedArray(imageL, whitemask)
    imageAM = np.ma.MaskedArray(imageA, whitemask)
    imageBM = np.ma.MaskedArray(imageB, whitemask)
    epsilon = 1e-11
    imageLMean = imageLM.mean()
    imageLSTD = imageLM.std() + epsilon
    imageAMean = imageAM.mean()
    imageASTD = imageAM.std() + epsilon
    imageBMean = imageBM.mean()
    imageBSTD = imageBM.std() + epsilon
    imageL = (imageL - imageLMean) / imageLSTD * target[0][1] + target[0][0]
    imageA = (imageA - imageAMean) / imageASTD * target[1][1] + target[1][0]
    imageB = (imageB - imageBMean) / imageBSTD * target[2][1] + target[2][0]
    imagelab = np.zeros(image.shape)
    imagelab[:, :, 0] = imageL
    imagelab[:, :, 1] = imageA
    imagelab[:, :, 2] = imageB
    returnimage = color.lab2rgb(imagelab)
    returnimage = np.clip(returnimage, 0, 1)
    returnimage *= 255
    returnimage[whitemask] = image[whitemask]
    return returnimage.astype(np.uint8)
