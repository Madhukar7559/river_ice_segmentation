import os, glob
import cv2
import sys
import numpy as np
import random
import imutils
from densenet.utils import linux_path, sort_key

import paramparse


#
# params = {
#     'db_root_dir': '/home/abhineet/N/Datasets/617/',
#     'seq_name': 'training',
#     'out_seq_name': '',
#     'fname_templ': 'img',
#     'img_ext': 'tif',
#     'out_ext': 'png',
#     'patch_height': 32,
#     'patch_width': 0,
#     'min_stride': 10,
#     'max_stride': 0,
#     'enable_flip': 0,
#     'enable_rot': 0,
#     'min_rot': 10,
#     'max_rot': 0,
#     'show_img': 0,
#     'n_frames': 0,
#     'start_id': 0,
#     'end_id': -1,
#     'enable_labels': 1
# }
# paramparse.from_dict(params, to_clipboard=True)
# exit()


class Params:
    def __init__(self):
        self.cfg = ()

        self.enable_labels = 1
        self.n_classes = 3
        self.proc_labels = 0
        self.allow_missing_labels = 1

        self.enable_flip = 0
        self.enable_rot = 0
        self.max_rot = 0
        self.max_stride = 0
        self.min_rot = 10
        self.min_stride = 10
        self.n_frames = 0
        self.patch_height = 32
        self.patch_width = 0

        self.seq_name = 'training'
        self.show_img = 0
        self.start_id = 0
        self.end_id = -1

        self.img_ext = 'tif'
        self.labels_ext = 'jpg'

        self.out_seq_name = ''
        self.out_img_ext = 'jpg'
        self.out_labels_ext = 'png'
        self.vis_ext = 'jpg'

        self.db_root_dir = ''
        self.src_path = ''
        self.labels_path = ''

        self.save_vis = 1


def get_vis_image(src_img, labels_img, n_classes, out_fname):
    labels_patch_ud_vis = labels_img * (255 / n_classes)
    if len(labels_patch_ud_vis.shape) == 1:
        labels_patch_ud_vis = cv2.cvtColor(labels_patch_ud_vis, cv2.COLOR_GRAY2BGR)
    labels_patch_ud_vis = np.concatenate((src_img, labels_patch_ud_vis), axis=1)
    cv2.imwrite(out_fname, labels_patch_ud_vis)

    return labels_patch_ud_vis


def run(params):
    """

    :param Params params:
    :return:
    """
    db_root_dir = params.db_root_dir
    seq_name = params.seq_name
    out_seq_name = params.out_seq_name
    img_ext = params.img_ext
    out_ext = params.out_labels_ext
    vis_ext = params.vis_ext
    show_img = params.show_img
    _patch_height = params.patch_height
    _patch_width = params.patch_width
    min_stride = params.min_stride
    max_stride = params.max_stride
    enable_flip = params.enable_flip
    enable_rot = params.enable_rot
    min_rot = params.min_rot
    max_rot = params.max_rot
    n_frames = params.n_frames
    start_id = params.start_id
    end_id = params.end_id
    enable_labels = params.enable_labels
    allow_missing_labels = params.allow_missing_labels
    n_classes = params.n_classes
    proc_labels = params.proc_labels

    src_path = params.src_path
    labels_path = params.labels_path

    if enable_labels:
        if not labels_path:
            labels_path = linux_path(db_root_dir, seq_name, 'labels')
        if not os.path.isdir(labels_path):
            print('Labels folder does not exist so disabling it')
            enable_labels = 0
        else:
            labels_path_root_dir = os.path.dirname(labels_path)

    if enable_rot and not enable_labels:
        raise SystemError('Rotation cannot be enabled without labels')

    if not src_path:
        src_path = linux_path(db_root_dir, seq_name, 'images')

    src_path_root_dir = os.path.dirname(src_path)

    print('Reading source images from: {}'.format(src_path))
    if enable_rot:
        print('rot: {:d}, {:d}'.format(min_rot, max_rot))

    src_files = [k for k in os.listdir(src_path) if k.endswith('.{:s}'.format(img_ext))]
    assert src_files, SystemError('No input frames found')
    total_frames = len(src_files)
    print('total_frames: {}'.format(total_frames))
    src_files.sort(key=sort_key)

    # src_file_list = src_file_list.sort()

    if n_frames <= 0:
        n_frames = total_frames

    if end_id < start_id:
        end_id = n_frames - 1

    patch_width, patch_height = _patch_width, _patch_height

    if patch_width <= 0:
        patch_width = patch_height

    if min_stride <= 0:
        min_stride = patch_height

    if max_stride <= min_stride:
        max_stride = min_stride

    image_as_patch = 0
    if _patch_width <= 0 and _patch_height <= 0:
        image_as_patch = 1
        print('Using entire image as the patch')

    if not out_seq_name:
        if image_as_patch:
            out_seq_name = '{:s}_{:d}_{:d}'.format(seq_name, start_id, end_id)
        else:
            out_seq_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                seq_name, start_id, end_id, patch_height, patch_width, min_stride, max_stride)

        if enable_rot:
            out_seq_name = '{}_rot_{:d}_{:d}'.format(out_seq_name, min_rot, max_rot)

        if enable_flip:
            out_seq_name = '{}_flip'.format(out_seq_name)

    if db_root_dir:
        out_src_path = linux_path(db_root_dir, out_seq_name, 'images')
    else:
        out_src_path = linux_path(src_path_root_dir, out_seq_name)

    print('Writing output images to: {}'.format(out_src_path))

    if not os.path.isdir(out_src_path):
        os.makedirs(out_src_path)

    if enable_labels:
        if db_root_dir:
            out_labels_path = linux_path(db_root_dir, out_seq_name, 'labels')
            out_labels_vis_path = linux_path(db_root_dir, out_seq_name, 'vis_labels')
        else:
            out_labels_path = linux_path(labels_path_root_dir, out_seq_name)
            out_labels_vis_path = linux_path(labels_path_root_dir, 'vis', out_seq_name)

        print('Writing output labels to: {}'.format(out_labels_path))
        os.makedirs(out_labels_path, exist_ok=1)

        print('Writing output visualization labels to: {}'.format(out_labels_vis_path))
        os.makedirs(out_labels_vis_path, exist_ok=1)

    rot_angle = 0
    n_out_frames = 0

    pause_after_frame = 1

    n_frames = end_id - start_id + 1

    for img_id in range(start_id, end_id + 1):

        # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
        img_fname = src_files[img_id]
        img_fname_no_ext, _ = os.path.splitext(img_fname)

        src_img_fname = linux_path(src_path, img_fname)
        src_img = cv2.imread(src_img_fname)

        src_height, src_width, _ = src_img.shape

        if image_as_patch:
            patch_width, patch_height = src_width, src_height

        if src_height < patch_height or src_width < patch_width:
            print('\nImage {} is too small {}x{} for the given patch size {}x{}\n'.format(
                src_img_fname, src_width, src_height, patch_width, patch_height))
            continue

        assert src_img is not None, 'Source image could not be read from: {}'.format(src_img_fname)

        if enable_labels:
            labels_img_fname = linux_path(labels_path, img_fname_no_ext + '.' + params.labels_ext)
            labels_img = cv2.imread(labels_img_fname)
            labels_img_orig = np.copy(labels_img)

            if labels_img is None:
                msg = 'Labels image could not be read from: {}'.format(labels_img_fname)
                if allow_missing_labels:
                    print('\n' + msg + '\n')
                    continue
                raise AssertionError(msg)

        if enable_rot:
            rot_angle = random.randint(min_rot, max_rot)
            src_img = imutils.rotate_bound(src_img, rot_angle)

            if not enable_labels:
                if params.save_vis:
                    out_src_img_fname = 'img_{:d}_rot_{:d}.{:s}'.format(
                        img_id + 1, rot_angle, vis_ext)

                    if db_root_dir:
                        out_src_img_dir = linux_path(db_root_dir, out_seq_name)
                    else:
                        out_src_img_dir = linux_path(src_path_root_dir, 'rot', out_seq_name)

                    os.makedirs(out_src_img_dir, exist_ok=1)

                    out_src_img_path = linux_path(out_src_img_dir, out_src_img_fname)
                    cv2.imwrite(out_src_img_path, src_img)
            else:
                if proc_labels:
                    if n_classes == 3:
                        labels_img[labels_img < 64] = 50
                        labels_img[np.logical_and(labels_img >= 64, labels_img < 192)] = 150
                        labels_img[labels_img >= 192] = 250
                    elif n_classes == 2:
                        labels_img[labels_img <= 128] = 50
                        labels_img[labels_img > 128] = 250
                    else:
                        raise AssertionError('unsupported number of classes: {}'.format(n_classes))
                else:
                    if n_classes == 3:
                        labels_img[labels_img == 0] = 50
                        labels_img[labels_img == 1] = 150
                        labels_img[labels_img == 2] = 250

                    elif n_classes == 2:
                        labels_img[labels_img == 0] = 50
                        labels_img[labels_img == 1] = 250

                labels_img = imutils.rotate_bound(labels_img, rot_angle).astype(np.int32)

                if params.save_vis:
                    out_labels_img_fname = 'labels_{:d}_rot_{:d}.{:s}'.format(
                        img_id + 1, rot_angle, vis_ext)

                    if db_root_dir:
                        out_labels_img_dir = linux_path(db_root_dir, out_seq_name)
                    else:
                        out_labels_img_dir = linux_path(labels_path_root_dir, 'rot', out_seq_name)

                    os.makedirs(out_labels_img_dir, exist_ok=1)

                    out_labels_img_path = linux_path(out_labels_img_dir, out_labels_img_fname)
                    labels_img_vis = np.concatenate((src_img, labels_img), axis=1)
                    cv2.imwrite(out_labels_img_path, labels_img_vis)

                labels_img[labels_img == 0] = -1

        n_rows, ncols, n_channels = src_img.shape

        if enable_labels:
            _n_rows, _ncols, _n_channels = labels_img.shape

            if n_rows != _n_rows or n_rows != _n_rows or n_rows != _n_rows:
                raise SystemError('Dimension mismatch between image and labels for file: {}'.format(img_fname))

            if proc_labels or enable_rot:
                if n_classes == 3:
                    labels_img[np.logical_and(labels_img >= 0, labels_img < 64)] = 0
                    labels_img[np.logical_and(labels_img >= 64, labels_img < 192)] = 1
                    labels_img[labels_img >= 192] = 2
                elif n_classes == 2:
                    labels_img[labels_img <= 128] = 0
                    labels_img[labels_img > 128] = 1
                else:
                    raise AssertionError('unsupported number of classes: {}'.format(n_classes))

        # np.savetxt(linux_path(db_root_dir, seq_name, 'labels_img_{}.txt'.format(img_id + 1)),
        #            labels_img[:, :, 2], fmt='%d')

        out_id = 0
        # skip_id = 0
        min_row = 0

        while True:
            max_row = min_row + patch_height
            if max_row > n_rows:
                diff = max_row - n_rows
                min_row -= diff
                max_row -= diff

            min_col = 0
            while True:
                max_col = min_col + patch_width
                if max_col > ncols:
                    diff = max_col - ncols
                    min_col -= diff
                    max_col -= diff

                if enable_labels:
                    labels_patch = labels_img[min_row:max_row, min_col:max_col, :]

                if enable_rot and (labels_patch == -1).any():
                    pass
                    # skip_id += 1
                    # sys.stdout.write('\rSkipping patch {:d}'.format(skip_id))
                else:
                    src_patch = src_img[min_row:max_row, min_col:max_col, :]

                    if image_as_patch:
                        out_img_fname = img_fname_no_ext
                    else:
                        out_img_fname = '{:s}_{:d}'.format(img_fname_no_ext, out_id + 1)

                    if enable_rot:
                        labels_patch = labels_patch.astype(np.uint8)
                        out_img_fname = '{:s}_rot_{:d}'.format(out_img_fname, rot_angle)

                    out_id += 1

                    out_src_img_fname = linux_path(out_src_path, '{:s}.{:s}'.format(out_img_fname, out_ext))
                    cv2.imwrite(out_src_img_fname, src_patch)

                    if enable_labels:
                        out_labels_img_fname = linux_path(out_labels_path, '{:s}.{:s}'.format(out_img_fname, out_ext))
                        cv2.imwrite(out_labels_img_fname, labels_patch)

                        if params.save_vis:
                            out_vis_labels_img_fname = linux_path(out_labels_vis_path,
                                                                  '{:s}.{:s}'.format(out_img_fname, vis_ext))
                            get_vis_image(src_patch, labels_patch, n_classes, out_vis_labels_img_fname)

                    if enable_flip:
                        src_patch_lr = np.fliplr(src_patch)
                        out_src_img_fname = linux_path(out_src_path, '{:s}_lr.{:s}'.format(out_img_fname, out_ext))
                        cv2.imwrite(out_src_img_fname, src_patch_lr)

                        src_patch_ud = np.flipud(src_patch)
                        out_src_img_fname = linux_path(out_src_path, '{:s}_ud.{:s}'.format(out_img_fname, out_ext))
                        cv2.imwrite(out_src_img_fname, src_patch_ud)

                        if enable_labels:
                            """
                            LR flip
                            """
                            labels_patch_lr = np.fliplr(labels_patch)
                            out_labels_img_fname = linux_path(out_labels_path,
                                                              '{:s}_lr.{:s}'.format(out_img_fname, out_ext))
                            cv2.imwrite(out_labels_img_fname, labels_patch_lr)

                            if params.save_vis:
                                out_vis_labels_img_fname = linux_path(out_labels_vis_path,
                                                                      '{:s}_lr.{:s}'.format(out_img_fname, vis_ext))
                                get_vis_image(src_patch_lr, labels_patch_lr, n_classes, out_vis_labels_img_fname)

                            """
                            UD flip
                            """
                            labels_patch_ud = np.flipud(labels_patch)
                            out_labels_img_fname = linux_path(out_labels_path,
                                                              '{:s}_ud.{:s}'.format(out_img_fname, out_ext))
                            cv2.imwrite(out_labels_img_fname, labels_patch_ud)

                            if params.save_vis:
                                out_vis_labels_img_fname = linux_path(out_labels_vis_path,
                                                                      '{:s}_ud.{:s}'.format(out_img_fname, vis_ext))
                                get_vis_image(src_patch_ud, labels_patch_ud, n_classes, out_vis_labels_img_fname)

                    if show_img:
                        disp_img = src_img.copy()
                        cv2.rectangle(disp_img, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)

                        # disp_labels_img = labels_img.copy()
                        # cv2.rectangle(disp_labels_img, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)

                        cv2.imshow('src_img', disp_img)
                        cv2.imshow('patch', src_patch)
                        if enable_labels:
                            labels_patch_disp = labels_img_orig[min_row:max_row, min_col:max_col, :].astype(np.uint8)
                            cv2.imshow('patch_labels', labels_patch_disp)
                        # cv2.imshow('disp_labels_img', disp_labels_img)
                        k = cv2.waitKey(1 - pause_after_frame)
                        if k == 27:
                            sys.exit(0)
                        elif k == 32:
                            pause_after_frame = 1 - pause_after_frame

                min_col += random.randint(min_stride, max_stride)
                if image_as_patch or max_col >= ncols:
                    break

            if image_as_patch or max_row >= n_rows:
                break
            min_row += random.randint(min_stride, max_stride)

        # sys.stdout.write('\nDone {:d}/{:d} frames with {:d} patches\n'.format(
        #     img_id + 1, n_frames, out_id))
        # sys.stdout.flush()
        n_out_frames += out_id
        sys.stdout.write('\rDone {:d}/{:d} frames'.format(img_id + 1, n_frames))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
    sys.stdout.write('Total frames generated: {}\n'.format(n_out_frames))


if __name__ == '__main__':
    _params = Params()

    paramparse.process(_params)

    run(_params)
