import os, time, sys
import numpy as np
import cv2



def processArguments(args, params):
    # arguments specified as 'arg_name=argv_val'
    no_of_args = len(args)
    for arg_id in range(no_of_args):
        arg = args[arg_id].split('=')
        if len(arg) != 2 or not arg[0] in params.keys():
            print('Invalid argument provided: {:s}'.format(args[arg_id]))
            return

        if not arg[1] or not arg[0] or arg[1] == '#':
            continue

        if arg[0].startswith('--'):
            arg[0] = arg[0][2:]

        if isinstance(params[arg[0]], (list, tuple)):
            # if not ',' in arg[1]:
            #     print('Invalid argument provided for list: {:s}'.format(arg[1]))
            #     return

            if arg[1] and ',' not in arg[1]:
                arg[1] = '{},'.format(arg[1])

            arg_vals = arg[1].split(',')
            arg_vals_parsed = []
            for _val in arg_vals:
                try:
                    _val_parsed = int(_val)
                except ValueError:
                    try:
                        _val_parsed = float(_val)
                    except ValueError:
                        _val_parsed = _val if _val else None

                if _val_parsed is not None:
                    arg_vals_parsed.append(_val_parsed)
            params[arg[0]] = arg_vals_parsed
        else:
            params[arg[0]] = type(params[arg[0]])(arg[1])


class LogWriter:
    def __init__(self, fname):
        self.fname=fname

    def _print(self, _str):
        print(_str + '\n')
        with open(self.fname, 'a') as fid:
            fid.write(_str + '\n')

def print_and_write(_str, fname=None):
    sys.stdout.write(_str + '\n')
    sys.stdout.flush()
    if fname is not None:
        open(fname, 'a').write(_str + '\n')


def sortKey(fname):
    fname = os.path.splitext(fname)[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)
    nums = [int(s) for s in fname.split('_') if s.isdigit()]
    non_nums = [s for s in fname.split('_') if not s.isdigit()]
    key = ''
    for non_num in non_nums:
        if not key:
            key = non_num
        else:
            key = '{}_{}'.format(key, non_num)
    for num in nums:
        if not key:
            key = '{:08d}'.format(num)
        else:
            key = '{}_{:08d}'.format(key, num)

    # try:
    #     key = nums[-1]
    # except IndexError:
    #     return fname

    # print('key: ', key)
    return key


def resizeAR(src_img, width, height, return_factors=False):
    aspect_ratio = float(width) / float(height)

    src_height, src_width, n_channels = src_img.shape
    src_aspect_ratio = float(src_width) / float(src_height)

    if src_aspect_ratio == aspect_ratio:
        dst_width = src_width
        dst_height = src_height
        start_row = start_col = 0
    elif src_aspect_ratio > aspect_ratio:
        dst_width = src_width
        dst_height = int(src_width / aspect_ratio)
        start_row = int((dst_height - src_height) / 2.0)
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, :] = src_img
    dst_img = cv2.resize(dst_img, (width, height))
    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, resize_factor, start_row, start_col
    else:
        return dst_img


def readData(images_path='', images_ext='', labels_path='', labels_ext='',
             images_type='source', labels_type='labels'):
    src_file_list = src_labels_list = None
    total_frames = 0

    if images_path and images_ext:
        print('Reading {} images from: {}'.format(images_type, images_path))
        src_file_list = [k for k in os.listdir(images_path) if k.endswith('.{:s}'.format(images_ext))]
        total_frames = len(src_file_list)
        if total_frames <= 0:
            raise SystemError('No input frames found')

        print('total_frames: {}'.format(total_frames))
        src_file_list.sort(key=sortKey)

    if labels_path and labels_ext:
        print('Reading {} images from: {}'.format(labels_type, labels_path))
        src_labels_list = [k for k in os.listdir(labels_path) if k.endswith('.{:s}'.format(labels_ext))]
        if src_file_list is not None:
            if total_frames != len(src_labels_list):
                raise SystemError('Mismatch between no. of labels and images')
        else:
            total_frames = len(src_labels_list)

        src_labels_list.sort(key=sortKey)

    return src_file_list, src_labels_list, total_frames


def getDateTime():
    return time.strftime("%y%m%d_%H%M", time.localtime())
