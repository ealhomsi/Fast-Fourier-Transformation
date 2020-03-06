import numpy as np
import argparse
import matplotlib
from dft import DFT

def __main__():
    results = None
    try:
        results = parseArgs()
    except BaseException as e:
        print("ERROR\tIncorrect input syntax: Please check arguments and try again")
        return
    
    mode = results.mode
    image = results.image


    DFT.test()

    if mode == 1:
        pass
    elif mode == 2:
        pass
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