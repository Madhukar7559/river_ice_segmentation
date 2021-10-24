import numpy as np
import cv2


def undo_resize_ar(resized_img, src_width, src_height, placement_type=0):
    height, width = resized_img.shape[:2]
    src_aspect_ratio = float(src_width) / float(src_height)
    aspect_ratio = float(width) / float(height)

    if src_aspect_ratio == aspect_ratio:
        dst_width = src_width
        dst_height = src_height
        start_row = start_col = 0
    elif src_aspect_ratio > aspect_ratio:
        dst_width = src_width
        dst_height = int(src_width / aspect_ratio)
        start_row = int((dst_height - src_height) / 2.0)
        if placement_type == 0:
            start_row = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        if placement_type == 0:
            start_col = 0
        elif placement_type == 1:
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_col = int(dst_width - src_width)
        start_row = 0

    height_resize_factor = float(dst_height) / float(height)
    width_resize_factor = float(dst_width) / float(width)

    assert height_resize_factor == width_resize_factor, "mismatch between height and width resize_factors"
    resized_img = resized_img.astype(np.uint8)
    unscaled_img = cv2.resize(resized_img, (dst_width, dst_height))
    unpadded_img = unscaled_img[start_row:start_row + src_height, start_col:start_col + src_width, ...]

    unpadded_img_disp, _, _ = resize_ar(unpadded_img, 1280)

    print('width, height: {}'.format((width, height)))
    print('dst_width, dst_height: {}'.format((dst_width, dst_height)))
    print('src_width, src_height: {}'.format((src_width, src_height)))

    # cv2.imshow('resized_img', resized_img)
    # cv2.imshow('unpadded_img', unpadded_img_disp)
    # cv2.waitKey(0)

    return unpadded_img


def resize_ar(src_img, width=0, height=0, placement_type=0, bkg_col=None):
    """
    resize an image to the given size while maintaining its aspect ratio and adding a black border as needed;
    if either width or height is omitted or specified as 0, it will be automatically computed from the other one;
    both width and height cannot be omitted;

    :param src_img: image to be resized
    :type src_img: np.ndarray

    :param width: desired width of the resized image; will be computed from height if omitted
    :type width: int

    :param height: desired height of the resized image; will be computed from width if omitted
    :type height: int

    :param return_factors: return the multiplicative resizing factor along with the
    position of the source image within the returned image with borders
    :type height: int

    :param placement_type: specifies how the source image is to be placed within the returned image if
    borders need to be added to achieve the target size;
    0: source image is top-left justified;
    1: source image is center-middle justified;
    2: source image is bottom-right justified;
    :type placement_type: int

    :return: resized image with optional resizing / placement info
    :rtype: np.ndarray | tuple[np.ndarray, float, int, int]
    """
    src_height, src_width = src_img.shape[:2]

    try:
        n_channels = src_img.shape[2]
    except IndexError:
        n_channels = 1

    src_aspect_ratio = float(src_width) / float(src_height)

    assert width > 0 or height > 0, 'Both width and height cannot be zero'

    if height <= 0:
        height = int(width / src_aspect_ratio)
    elif width <= 0:
        width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    if src_aspect_ratio == aspect_ratio:
        dst_width = src_width
        dst_height = src_height
        start_row = start_col = 0
    elif src_aspect_ratio > aspect_ratio:
        dst_width = src_width
        dst_height = int(src_width / aspect_ratio)
        start_row = int((dst_height - src_height) / 2.0)
        if placement_type == 0:
            start_row = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        if placement_type == 0:
            start_col = 0
        elif placement_type == 1:
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_col = int(dst_width - src_width)
        start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8).squeeze()

    if bkg_col is not None:
        dst_img.fill(bkg_col)

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, ...] = src_img
    dst_img = cv2.resize(dst_img, (width, height))

    resize_factor = float(height) / float(dst_height)

    img_bbox = [start_col, start_row, start_col + src_width, start_row + src_height]
    return dst_img, resize_factor, img_bbox
