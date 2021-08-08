import sys
import os
import shlex
import subprocess
from datetime import datetime

import paramparse
from paramparse import MultiPath

from densenet.utils import linux_path


#
# params = {
#     'db_root_dir': '/home/abhineet/N/Datasets/617/',
#     'seq_name': 'training',
#     'fname_templ': 'img',
#     'img_ext': 'tif',
#     'out_ext': 'png',
#     'patch_height': 32,
#     'patch_width': 0,
#     'min_stride': 10,
#     'max_stride': 0,
#     'enable_flip': 0,
#     'min_rot': 15,
#     'max_rot': 345,
#     'n_rot': 3,
#     'show_img': 0,
#     'n_frames': 0,
#     'start_id': 0,
#     'end_id': -1,
# }
#
# paramparse.from_dict(params, to_clipboard=1)
# exit()


class Params:
    def __init__(self):
        self.cfg_root = 'cfg'
        self.cfg_ext = 'cfg'
        self.cfg = ()

        self.dataset = ''
        self.seq_name = MultiPath()
        self.image_dir = 'Images'
        self.labels_dir = 'Labels'
        self.seq_name = MultiPath()
        self.n_classes = 3

        self.db_root_dir = ''
        self.src_path = ''
        self.labels_path = ''

        self.img_ext = 'tif'
        self.labels_ext = 'jpg'
        self.out_img_ext = 'jpg'
        self.out_labels_ext = 'png'

        self.start_id = 0
        self.end_id = -1

        self.enable_flip = 0
        self.max_rot = 345
        self.max_stride = 0
        self.min_rot = 15
        self.min_stride = 10
        self.n_frames = 0
        self.n_rot = 3
        self.patch_height = 32
        self.patch_width = 0
        self.show_img = 0
        self.parallel = 1
        self.log_to_file = 0
        self.log_dir = 'log'


def run(params):
    """

    :param Params params:
    :return:
    """
    db_root_dir = params.db_root_dir
    seq_name = params.seq_name
    img_ext = params.img_ext
    labels_ext = params.labels_ext
    out_img_ext = params.out_img_ext
    out_labels_ext = params.out_labels_ext
    show_img = params.show_img
    patch_height = params.patch_height
    patch_width = params.patch_width
    min_stride = params.min_stride
    max_stride = params.max_stride
    enable_flip = params.enable_flip
    min_rot = params.min_rot
    max_rot = params.max_rot
    n_rot = params.n_rot
    n_frames = params.n_frames
    start_id = params.start_id
    end_id = params.end_id
    src_path = params.src_path
    labels_path = params.labels_path
    n_classes = params.n_classes

    if not src_path:
        src_path = linux_path(db_root_dir, seq_name, 'images')

    if not labels_path:
        labels_path = os.path.join(db_root_dir, seq_name, 'labels')

    src_files = [k for k in os.listdir(src_path) if k.endswith('.{:s}'.format(img_ext))]
    total_frames = len(src_files)
    # print('file_list: {}'.format(file_list))
    if total_frames <= 0:
        raise SystemError('No input frames found')
    print('total_frames: {}'.format(total_frames))

    if n_frames <= 0:
        n_frames = total_frames

    if end_id < start_id:
        end_id = n_frames - 1

    if patch_width <= 0:
        patch_width = patch_height

    if min_stride <= 0:
        min_stride = patch_height

    if max_stride <= min_stride:
        max_stride = min_stride

    rot_range = int(float(max_rot - min_rot) / float(n_rot))

    out_seq_names = []

    cmb_out_seq_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_rot_{:d}_{:d}_{:d}'.format(
        seq_name, start_id, end_id, patch_height, patch_width, min_stride, max_stride, min_rot, max_rot, n_rot)

    if enable_flip:
        cmb_out_seq_name = '{}_flip'.format(cmb_out_seq_name)

    base_cmd = 'python3 subPatchDataset.py db_root_dir={} seq_name={}' \
               ' img_ext={} labels_ext={} out_img_ext={} out_labels_ext={} ' \
               'patch_height={} patch_width={} min_stride={} max_stride={} enable_flip={} start_id={} end_id={} ' \
               'n_frames={} show_img={} out_seq_name={} src_path={} labels_path={} n_classes={}'.format(
        db_root_dir, seq_name, img_ext, labels_ext, out_img_ext, out_labels_ext, patch_height, patch_width, min_stride,
        max_stride,
        enable_flip, start_id, end_id, n_frames, show_img, cmb_out_seq_name, src_path, labels_path, n_classes)

    # out_seq_name_base = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
    #     seq_name, start_id, end_id, patch_height, patch_width, min_stride, max_stride)
    # out_seq_name = out_seq_name_base
    # if enable_flip:
    #     out_seq_name = '{}_flip'.format(out_seq_name_base)
    # out_seq_names.append(out_seq_name)

    processes = []
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    tee_log_id = 'sub_patch_batch_{}'.format(time_stamp)
    os.makedirs(params.log_dir, exist_ok=True)

    no_rot_cmd = '{} enable_rot=0'.format(base_cmd)
    print('\n\nRunning:\n {}\n\n'.format(no_rot_cmd))

    if params.parallel:
        args = shlex.split(no_rot_cmd)
        p = subprocess.Popen(args)
        if params.log_to_file:
            out_fname = tee_log_id + '.ansi'
            zip_fname = out_fname.replace('.ansi', '.zip')

            out_path = linux_path(params.log_dir, out_fname)

            f = open(out_path, 'w')
            p = subprocess.Popen(args, stdout=f, stderr=f)
        else:
            f = out_fname = zip_fname = None
            p = subprocess.Popen(args)

        processes.append((p, f, out_fname, zip_fname))
    else:
        subprocess.check_call(no_rot_cmd, shell=True)

    _min_rot = min_rot

    for i in range(n_rot):
        if i == n_rot - 1:
            _max_rot = max_rot
        else:
            _max_rot = _min_rot + rot_range

        rot_cmd = '{} enable_rot=1 min_rot={} max_rot={}'.format(base_cmd, _min_rot, _max_rot)
        print('\n\nrot {} / {}:\n {}\n\n'.format(i, n_rot, rot_cmd))

        if params.parallel:
            args = shlex.split(rot_cmd)
            p = subprocess.Popen(args)
            if params.log_to_file:
                out_fname = tee_log_id + '_{}.ansi'.format(i)
                zip_fname = out_fname.replace('.ansi', '.zip')

                out_path = linux_path(params.log_dir, out_fname)
                # zip_path = os.path.join(params.log_dir, zip_fname)

                f = open(out_path, 'w')
                p = subprocess.Popen(args, stdout=f, stderr=f)
            else:
                f = out_fname = zip_fname = None
                p = subprocess.Popen(args)

            processes.append((p, f, out_fname, zip_fname))
        else:
            subprocess.check_call(rot_cmd, shell=True)

        # out_seq_name = '{}_rot_{:d}_{:d}'.format(out_seq_name_base, min_rot, max_rot)
        # if enable_flip:
        #     out_seq_name = '{}_flip'.format(out_seq_name_base)
        # out_seq_names.append(out_seq_name)

        # cmb_out_seq_name_base = '{}_{}'.format(cmb_out_seq_name_base, _min_rot)
        _min_rot = _max_rot + 1

        # merge_base_cmb = 'python3 mergeDatasets.py'
        # for i in range(len(out_seq_names)):
        #     merge_cmd = '{} {} {} start_id={} end_id={}'.format(
        #         merge_base_cmb, out_seq_names[i], cmb_out_seq_name, start_id, end_id)
        #     print('\n\nRunning:\n {}\n\n'.format(merge_cmd))
        #     subprocess.check_call(merge_cmd, shell=True)

    if params.parallel:
        for p, f, f_name, zip_fname in processes:
            p.wait()
            if f is None:
                continue

            f.close()


if __name__ == '__main__':
    _params = Params()

    paramparse.process(_params)

    if _params.dataset == 'ctc':
        from new_deeplab.datasets.build_ctc_data import CTCInfo

        db_splits = CTCInfo.DBSplits().__dict__

        split = _params.seq_name
        db_root_dir = _params.db_root_dir

        _params.db_root_dir = ''

        seq_ids = db_splits[split]
        for seq_id in seq_ids:
            seq_name, n_frames = CTCInfo.sequences[seq_id]
            _params.seq_name = seq_name

            _params.src_path = linux_path(db_root_dir, _params.image_dir, seq_name)
            _params.labels_path = linux_path(db_root_dir, _params.labels_dir, seq_name)

            run(_params)
    else:
        run(_params)
