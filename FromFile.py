#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

import argparse
import logging
import numpy as np
import cv2

import skin_detector

logger = logging.getLogger('main')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_paths', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-b', '--debug', dest='debug', action='store_true', help='enable debug logging')
    parser.add_argument('-q', '--quite', dest='quite', action='store_true', help='disable all logging')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help="display result")
    parser.add_argument('-s', '--save', dest='save', action='store_true', help="save result to file")
    parser.add_argument('-t', '--thresh', dest='thresh', default=0.5, type=float, help='threshold for skin mask')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    for image_arg in args.image_paths:
        for image_path in skin_detector.find_images(image_arg):
            logging.info("loading image from {0}".format(image_path))
            img_col = cv2.imread(image_path, 1)
            img_copy = img_col.copy()
            
            img_msk = skin_detector.process(img_col)

            skin=cv2.blur(img_copy, (27, 27))
            skin_extract=cv2.bitwise_and(skin, skin, mask=img_msk)
            

            mask_back=cv2.bitwise_not(img_msk)
            back=cv2.bitwise_and(img_col, img_col, mask=mask_back)

            final = cv2.add(back, skin_extract)

            if args.display:
                skin_detector.scripts.display('img_col', img_col)
                skin_detector.scripts.display('img_msk', img_msk)
                skin_detector.scripts.display('skin_extract', skin_extract)
                #skin_detector.scripts.display('mask_back', mask_back)
                # skin_detector.scripts.display('back', back)
                skin_detector.scripts.display('final', final)
                cv2.waitKey(0)
