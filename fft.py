import argparse
import math

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg

import numpy as np

from dft import DFT


def desiredSize(n):
    p = int(math.log(n, 2)); 
    return int(pow(2, p+1));  

def __main__():
    results = None
    try:
        results = parseArgs()
    except BaseException as e:
        print("ERROR\tIncorrect input syntax: Please check arguments and try again")
        return
    
    mode = results.mode
    image = results.image


    # run tests
    DFT.test()

    if mode == 1:
        # read the image
        im_raw = plt.imread(image).astype(float)

        # pad the image to desired size
        old_shape = im_raw.shape
        new_shape = desiredSize(old_shape[0]), desiredSize(old_shape[1])
        im = np.zeros(new_shape) 
        im[:old_shape[0], :old_shape[1]] = im_raw

        # perform fft 2d
        fft_im = DFT.fast_two_dimension(im)
        
        #display
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im, plt.cm.gray)
        ax[0].set_title('original')
        ax[1].imshow(np.abs(fft_im), norm=colors.LogNorm())
        ax[1].set_title('fft 2d with lognorm')
        fig.suptitle('Mode 1')
        plt.show()

    elif mode == 2:
        # define a percentage keep fraction
        keep_fraction = 0.08

        # read the image
        im_raw = plt.imread(image).astype(float)

        # pad the image to desired size
        old_shape = im_raw.shape
        new_shape = desiredSize(old_shape[0]), desiredSize(old_shape[1])
        im = np.zeros(new_shape) 
        im[:old_shape[0], :old_shape[1]] = im_raw

        # perform fft 2d and remove high frequency values
        fft_im = DFT.fast_two_dimension(im)
        r,c = fft_im.shape
        print("Fraction of pixels used {} and the number is ({}, {}) out of ({}, {})".format(keep_fraction, int(keep_fraction*r), int(keep_fraction*c), r, c))
        fft_im[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
        fft_im[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
        
        # perform ifft 2d to denoise the image
        denoised = DFT.fast_two_dimension_inverse(fft_im).real

        #display
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im, plt.cm.gray)
        ax[0].set_title('original')
        ax[1].imshow(denoised, plt.cm.gray)
        ax[1].set_title('denoised')
        fig.suptitle('Mode 2')
        plt.show()
    elif mode == 3:
        pass
    elif mode == 4:
        pass
    else:
        print("ERROR\tMode {} is not recofgnized".format(mode))
        return



def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', dest='mode',
                        help='Mode of operation 1-> fast, 2-> denoise, 3-> compress&save 4-> plot', type=int, default=1)
    parser.add_argument('-i', action='store', dest='image',
                        help='image path to work on', type=str, default='moonlanding.png')
    return parser.parse_args()

if __name__ == "__main__":
    __main__()
