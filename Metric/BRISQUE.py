# Example run: python -W ignore BRISQUE.py --test_dir /root/autodl-tmp/Result/RetinexNet/ExDark --read_subfolder True

import os
from glob import glob
from PIL import Image
import numpy as np
from skimage import io, img_as_float
import imquality.brisque as brisque
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_dir', type=str,
                        help='directory for testing inputs')
    parser.add_argument('--read_subfolder', type=bool, default=False)                        
    args = parser.parse_args()
    return args

def cal_loe(inp):
    pass


def cal_brisque(inp, i, AssertionError_count):
    imgOri = Image.open(inp)
    img = img_as_float(imgOri)
    try:
        score = brisque.score(img)
    except AssertionError:
        score = 0
        AssertionError_count += 1
        print(i, " path[i] is", inp)
    return score, AssertionError_count


def main(args):
    if args.read_subfolder:
        path = glob(os.path.join(args.test_dir, '*/*'))
    else:
        path = glob(os.path.join(args.test_dir, '*'))

    list_brisque = []

    AssertionError_count = 0
    for i in range(len(path)):
    # for i in tqdm(range(1550, len(path))):

        # calculate scores
        # print(i, " path[i] is", path[i])
        score, AssertionError_count = cal_brisque(path[i], i, AssertionError_count)
        if score != 0:
        # append to list
            list_brisque.append(score)

    # Average score for the dataset
    print("Have ", AssertionError_count, " times AssertionError.")
    print("Average BRISQUE:", "%.3f" % (np.mean(list_brisque)))



if __name__ == "__main__":
    args = parse_args()
    main(args)
