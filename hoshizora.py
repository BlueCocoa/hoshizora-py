#!/usr/bin/python3
# -*- coding: utf-8 -*-


from optparse import OptionParser

# support OpenCV / PIL
# OpenCV is first choice
using_cv = False
try:
    import cv2
    import numpy as np
    using_cv = True
except ImportError:
    try:
        from PIL import Image
    except ImportError:
        raise "OpenCV / PIL is required"


def parse_arg():
    """
    Parsing command line arguments
    
    Returns
    -------
    optparse.Values
        Parsed command line options
    """
    parser = OptionParser()
    parser.add_option("-f", "--front", dest="front", type=str, help="front layer")
    parser.add_option("-b", "--back", dest="back", type=str, help="back layer")
    parser.add_option("-o", "--output", dest="output", type=str, help="output image")
    parser.add_option("-i", "--increase", default=0, dest="increase", type=int, help="increase brightness of front layer")
    parser.add_option("-d", "--decrease", default=0, dest="decrease", type=int, help="decrease brightness of back layer")
    (options, args) = parser.parse_args()
    return options


def img_shape(frontlayer, backlayer):
    """
    Retrive Image Shape

    Parameters
    ----------
    frontlayer : numpy / PIL Image
        Image for the front layer
    backlayer : numpy / PIL Image
        Image for the back layer

    Returns
    -------
    Tuple
        ``f_rows``, ``f_cols``, ``b_rows``, ``b_cols``
    """
    global using_cv
    if using_cv:
        # OpenCV
        f_rows, f_cols = frontlayer.shape
        b_rows, b_cols = backlayer.shape
    else:
        # PIL
        f_cols, f_rows = frontlayer.size
        b_cols, b_rows = backlayer.size
    return f_rows, f_cols, b_rows, b_cols


def resize_layer(layer, rows : int, cols : int):
    """
    Actual Resize
    
    Parameters
    ----------
    layer : numpy / PIL Image
        Image to be resized
    rows : int
        Rows after resized
    cols : int
        Cols after resized
    
    Returns
    -------
    numpy / PIL Image
        Resized image
    """
    global using_cv
    if using_cv:
        return cv2.resize(layer, (rows, cols))
    else:
        return layer.resize((rows, cols))


def resize_layers(frontlayer, backlayer):
    """
    Coordinate resizing with both frontlayer and backlayer
    
    Parameters
    ----------
    frontlayer : numpy / PIL Image
        Image for the front layer
    backlayer : numpy / PIL Image
        Image for the back layer
    
    Returns
    -------
    Tuple
        Resized layers
    """
    f_rows, f_cols, b_rows, b_cols = img_shape(frontlayer, backlayer)
    # basic idea
    # 1. keep aspect ratio
    # 2. fit to large one (both width and height)
    # e.g,
    #   front:  500x800
    #   back:   700x400
    #   output: 700x800
    if f_cols > b_cols:
        if f_rows > b_rows:
            if (f_cols / f_rows) > (b_cols / b_rows):
                backlayer = resize_layer(backlayer, int(b_cols * f_rows / b_rows), f_rows)
            else:
                backlayer = resize_layer(backlayer, f_cols, int(b_rows * f_cols / b_cols))
        else:
            backlayer = resize_layer(backlayer, int(b_cols * f_rows / b_rows), f_rows)
    else:
        if f_rows < b_rows:
            if (f_cols / f_rows) > (b_cols / b_rows):
                frontlayer = resize_layer(frontlayer, int(f_cols * b_rows / f_rows), b_rows)
            else:
                frontlayer = resize_layer(frontlayer, int(f_rows * b_rows / f_cols), b_cols)
        else:
            frontlayer = resize_layer(frontlayer, int(f_cols * b_rows / f_rows), b_rows)
    return frontlayer, backlayer


def load_layers(options):
    """
    Load images in grey based on command line options
    
    Parameters
    ----------
    options : optparse.Values
        Parsed command line options
    
    Returns
    -------
    Tuple
        ``frontlayer``, ``backlayer``
    """
    global using_cv
    if using_cv:
        # read and convert to grey image
        frontlayer = cv2.imread(options.front, 0)
        backlayer = cv2.imread(options.back, 0)
    else:
        # read and convert to grey image
        frontlayer = Image.open(options.front).convert("L")
        backlayer = Image.open(options.back).convert("L")
    return frontlayer, backlayer


def overlay_center(canvas, image):
    """
    Overlay original image onto canvas
    
    Parameters
    ----------
    canvas : numpy / PIL Image
        Corresponding canvas for the image
    image : numpy / PIL Image
        Either frontlayer or backlayer
    """
    global using_cv
    if using_cv:
        # get upper left point
        canvas_size = canvas.shape[:2]
        overlay_rows_start = (canvas_size[0] - image.shape[0]) // 2
        overlay_cols_start = (canvas_size[1] - image.shape[1]) // 2
        # overlay original image
        canvas[overlay_rows_start:overlay_rows_start+image.shape[0], overlay_cols_start:overlay_cols_start+image.shape[1]] = image
    else:
        # get upper left point
        canvas_size = canvas.size
        overlay_rows_start = (canvas_size[1] - image.size[1]) // 2
        overlay_cols_start = (canvas_size[0] - image.size[0]) // 2
        # overlay original image
        canvas.paste(image, (overlay_rows_start, overlay_cols_start))


def create_canvas(mix_size):
    """
    Create canvas
    
    Parameters
    ----------
    mix_size : tuple
        (rows, cols, channels)
    
    Returns
    -------
    Tuple
        White ``frontlayer``, black ``backlayer``
    """
    global using_cv
    if using_cv:
        front_canvas = np.ones(mix_size[:2], dtype=np.float) * 255
        back_canvas = np.zeros(mix_size[:2], dtype=np.float)
    else:
        front_canvas = Image.new("L", (mix_size[1], mix_size[0]), 255)
        back_canvas = Image.new("L", (mix_size[1], mix_size[0]))
    return front_canvas, back_canvas


def color_shift(options, front_canvas, back_canvas):
    """
    Shift color intensity
    
    Parameters
    ----------
    options : optparse.Values
        Parsed command line options
    frontlayer : numpy / PIL Image
        Image for the front layer
    backlayer : numpy / PIL Image
        Image for the back layer
        
    Returns
    -------
    Tuple
        Color intensity shifted ``frontlayer``, ``backlayer``
    """
    global using_cv
    if using_cv:
        # increase frontlayer
        front_canvas += options.increase
        front_canvas[front_canvas > 255] = 255
        # decrease backlayer
        back_canvas -= options.decrease
        back_canvas[back_canvas < 0] = 0
    else:
        # increase frontlayer
        front_canvas = front_canvas.point(lambda p: min(p + options.increase, 255))
        # decrease backlayer
        back_canvas = back_canvas.point(lambda p: max(p - options.decrease, 0))
    return front_canvas, back_canvas


def compute_alpha(front_canvas, back_canvas):
    """
    Compute alpha channel of output
    
    Parameters
    ----------
    front_canvas : numpy / PIL Image
        Image for the front layer
    back_canvas : numpy / PIL Image
        Image for the back layer

    Returns
    -------
    numpy / list
        Alpha channel of output
    """
    global using_cv
    if using_cv:
        A = back_canvas + 255 - front_canvas
        A[A > 255] = 255
        A[A <= 0] = 1e-12
        return A
    else:
        front_canvas_data = front_canvas.getdata()
        back_canvas_data = back_canvas.getdata()
        tmp = list(map(lambda p: p + 255, back_canvas_data))
        A = map(lambda i: max(1e-12, min(tmp[i] - front_canvas_data[i], 255)), range(len(back_canvas_data)))
        return list(A)


def compute_grey(back_canvas, alpha):
    """
    Compute grey channel of output
    
    Parameters
    ----------
    back_canvas : numpy / PIL Image
        Image for the back layer
    alpha : numpy / list
        Alpha channel of output

    Returns
    -------
    numpy / list
        Grey channel of output
    """
    global using_cv
    if using_cv:
        G = np.divide(back_canvas * 255, A)
        return G
    else:
        back_canvas_data = back_canvas.getdata()
        G = map(lambda i: back_canvas_data[i] * 255 / A[i], range(len(A)))
        return list(G)


def create_mix_canvas(mix_size, alpha, grey):
    """
    Create and merge alpha and grey channels into output canvas
    
    Parameters
    ----------
    mix_size : tuple
        (rows, cols, channels)
    alpha : numpy / list
        Alpha channel of output
    grey : numpy / list
        Grey channel of output
    
    Returns
    -------
    numpy / PIL Image
        Output image
    """
    global using_cv
    if using_cv:
        mix_canvas = np.zeros(mix_size, dtype=np.uint8)
        mix_canvas[:, :, 0] = grey
        mix_canvas[:, :, 1] = grey
        mix_canvas[:, :, 2] = grey
        mix_canvas[:, :, 3] = alpha
    else:
        # convert list data into PIL Image
        grey_image = Image.new("L", (mix_size[1], mix_size[0]))
        grey_image.putdata(grey)
        alpha_image = Image.new("L", (mix_size[1], mix_size[0]))
        alpha_image.putdata(alpha)
        # merge 4 channels
        mix_canvas = Image.merge("RGBA", (grey_image, grey_image, grey_image, alpha_image))
    return mix_canvas


def save_image(options, mix_canvas):
    """
    Save image
    
    Parameters
    ----------
    options : optparse.Values
        Parsed command line options
    mix_canvas : numpy / PIL Image
        Output image
    """
    global using_cv
    if using_cv:
        cv2.imwrite(options.output, mix_canvas)
    else:
        mix_canvas.save(options.output)


if __name__ == '__main__':
    options = parse_arg()
    frontlayer, backlayer = load_layers(options)
    frontlayer, backlayer = resize_layers(frontlayer, backlayer)
    f_rows, f_cols, b_rows, b_cols = img_shape(frontlayer, backlayer)
    mix_size = (max([f_rows, b_rows]), max([f_cols, b_cols]), 4)

    front_canvas, back_canvas = create_canvas(mix_size)
    overlay_center(front_canvas, frontlayer)
    overlay_center(back_canvas, backlayer)

    front_canvas, back_canvas = color_shift(options, front_canvas, back_canvas)
    A = compute_alpha(front_canvas, back_canvas)
    G = compute_grey(back_canvas, A)
    mix_canvas = create_mix_canvas(mix_size, A, G)
    save_image(options, mix_canvas)
