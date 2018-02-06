import cv2
import numpy as np
import os


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    return noisy_image

def main():
    for imagename in os.listdir("/hoem04/outofhome/MURA_TEST"): # TRAIN/TEST
        # cv2.IMREAD_COLOR - read in color images (BGR)
        # cv2.IMREAD_GRAYSCALE - convert image to grayscale
        imagenamepath1 = "/hoem04/outofhome/MURA_TEST/" + imagename # TRAIN/TEST
        image = cv2.imread(imagenamepath1, cv2.IMREAD_GRAYSCALE)
        #if len(image.shape) == 2:
        #    grayscale_image = cv2.cvt

        if image.shape[0] != 512 and image.shape[1] != 512:
            print("hell")
            exit(0)
        if image.shape[0] == 512:
            np.transpose(image)

        image = cv2.resize(image, (350, 512), interpolation = cv2.INTER_CUBIC)

        grayscale_image = image
        #grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        """
        Gaussian is a nice noise function.  Gaussian noise are values generated from the
        random normal distribution.  The mean of the distribution is 0 and the standard
        deviation is 1.  The standard deviation is a measure of how spread out the values
        are from the mean or 0.  randn() generates random numbers from this distribution.
        The Gaussian distribution is symmetric about the mean of the probability.
    
        Sigma determines the magnitude of the noise function.  For a small sigma, the noise
        function produces values very close to zero or a gray image since we want to map the
        pixel with a value of zero to gray.  The larger sigma spreads out the noise.
        Multiplying an image by a noise image generated from a Gaussian function effectively
        changes the standard deviation of the pixel values.  This is how far apart the pixel
        colors are in value.
        """
        noisy_sigma = 20
        noisy_image = add_gaussian_noise(grayscale_image, noisy_sigma)

        noisy_filename = imagename
        imagenamepath2 = "/hoem04/outofhome/MURA_TEST_RESIZE_NOISE/test/" + noisy_filename # TRAIN/TEST
        cv2.imwrite(imagenamepath2, noisy_image)

if __name__ == "__main__":
    main()
