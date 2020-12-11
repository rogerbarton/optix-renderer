from skimage.restoration import denoise_bilateral

from skimage import data, img_as_float
from skimage.io import imread, imsave

noisy = imread("noisy.exr")

print("Denoising...")

denoised = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=1, multichannel=True)

imsave("python.exr", denoised)