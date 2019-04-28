#/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from optparse import OptionParser


def parse_arg():
    parser = OptionParser()
    parser.add_option("-f", "--front", dest="front", type=str, help="front layer")
    parser.add_option("-b", "--back", dest="back", type=str, help="back layer")
    parser.add_option("-o", "--output", dest="output", type=str, help="output image")
    parser.add_option("-i", "--increase", dest="increase", type=int, help="increase brightness of front layer")
    parser.add_option("-d", "--decrease", dest="decrease", type=int, help="decrease brightness of back layer")
    (options, args) = parser.parse_args()
    return options


def resize_layers(frontlayer, backlayer):
    f_rows, f_cols = frontlayer.shape
    b_rows, b_cols = backlayer.shape
    if f_cols > b_cols:
        if f_rows > b_rows:
            if (f_cols / f_rows) > (b_cols / b_rows):
                backlayer = cv2.resize(backlayer, (int(b_cols * f_rows / b_rows), f_rows))
            else:
                backlayer = cv2.resize(backlayer, (f_cols, int(b_rows * f_cols / b_cols)))
        else:
            backlayer = cv2.resize(backlayer, (int(b_cols * f_rows / b_rows), f_rows))
    else:
        if f_rows < b_rows:
            if (f_cols / f_rows) > (b_cols / b_rows):
                frontlayer = cv2.resize(frontlayer, (int(f_cols * b_rows / f_rows), b_rows))
            else:
                frontlayer = cv2.resize(frontlayer, (int(f_rows * b_rows / f_cols), b_cols))
        else:
            frontlayer = cv2.resize(frontlayer, (int(f_cols * b_rows / f_rows), b_rows))
    return frontlayer, backlayer


def overlay_center(canvas, image):
    canvas_size = canvas.shape[:2]
    overlay_rows_start = (canvas_size[0] - image.shape[0]) // 2
    overlay_cols_start = (canvas_size[1] - image.shape[1]) // 2
    canvas[overlay_rows_start:overlay_rows_start+image.shape[0], overlay_cols_start:overlay_cols_start+image.shape[1]] = image

if __name__ == '__main__':
    options = parse_arg()
    frontlayer = cv2.imread(options.front, 0)
    backlayer = cv2.imread(options.back, 0)
    frontlayer, backlayer = resize_layers(frontlayer, backlayer)
    mix_size = (max([frontlayer.shape[0], backlayer.shape[0]]), max([frontlayer.shape[1], backlayer.shape[1]]), 4)

    front_canvas = np.ones(mix_size[:2], dtype=np.float) * 255
    back_canvas = np.zeros(mix_size[:2], dtype=np.float)
    overlay_center(front_canvas, frontlayer)
    overlay_center(back_canvas, backlayer)

    front_canvas += options.increase
    front_canvas[front_canvas > 255] = 255
    back_canvas -= options.decresase
    back_canvas[back_canvas < 0] = 0

    A = back_canvas + 255 - front_canvas
    A[A > 255] = 255
    A[A <= 0] = 1e-12
    G = np.divide(back_canvas * 255, A)
    mix_canvas = np.zeros(mix_size, dtype=np.uint8)

    mix_canvas[:, :, 0] = G
    mix_canvas[:, :, 1] = G
    mix_canvas[:, :, 2] = G
    mix_canvas[:, :, 3] = A
    cv2.imwrite(options.output, mix_canvas)
