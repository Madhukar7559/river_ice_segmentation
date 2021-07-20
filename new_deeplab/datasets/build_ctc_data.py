# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts CTC data to TFRecord file format with Example protos."""

import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import sys
import build_data
import cv2
import numpy as np

import tensorflow as tf

import paramparse

from tqdm import tqdm


def irange(a, b):
    return list(range(a, b + 1))


class CTCInfo:
    class DBSplits:
        def __init__(self):
            self.all_r = irange(0, 19)
            self.bf_r = irange(0, 3)
            self.bf1_r = irange(0, 1)
            self.bf2_r = irange(2, 3)
            self.dic_r = irange(4, 5)
            self.fluo_r = irange(6, 15)
            self.fluo1_r = irange(6, 11)
            self.fluo2_r = irange(12, 15)
            self.huh_r = irange(6, 7)
            self.gow_r = irange(8, 9)
            self.sim_r = irange(10, 11)
            self.hela_r = irange(14, 15)
            self.phc_r = irange(16, 19)
            self.phc1_r = irange(16, 17)
            self.phc2_r = irange(18, 19)

            self.all_e = irange(20, 39)
            self.bf_e = irange(20, 23)
            self.bf1_e = irange(20, 21)
            self.bf2_e = irange(22, 23)
            self.dic_e = irange(24, 25)
            self.fluo_e = irange(26, 35)
            self.fluo1_e = irange(26, 31)
            self.fluo2_e = irange(32, 35)
            self.huh_e = irange(26, 27)
            self.gow_e = irange(28, 29)
            self.sim_e = irange(30, 31)
            self.hela_e = irange(34, 35)
            self.phc_e = irange(36, 39)
            self.phc1_e = irange(36, 37)
            self.phc2_e = irange(38, 39)

            self.all = self.all_r + self.all_e
            self.bf = self.bf_r + self.bf_e
            self.bf1 = self.bf1_r + self.bf1_e
            self.bf2 = self.bf2_r + self.bf2_e
            self.dic = self.dic_r + self.dic_e
            self.fluo = self.fluo_r + self.fluo_e
            self.fluo1 = self.fluo1_r + self.fluo1_e
            self.fluo2 = self.fluo2_r + self.fluo2_e
            self.huh = self.huh_r + self.huh_e
            self.gow = self.gow_r + self.gow_e
            self.sim = self.sim_r + self.sim_e
            self.hela = self.hela_r + self.hela_e
            self.phc = self.phc_r + self.phc_e
            self.phc1 = self.phc1_r + self.phc1_e
            self.phc2 = self.phc2_r + self.phc2_e

    sequences = {
        # train
        0: ('BF-C2DL-HSC_01', 1764),
        1: ('BF-C2DL-HSC_02', 1764),
        2: ('BF-C2DL-MuSC_01', 1376),
        3: ('BF-C2DL-MuSC_02', 1376),

        4: ('DIC-C2DH-HeLa_01', 84),
        5: ('DIC-C2DH-HeLa_02', 84),

        6: ('Fluo-C2DL-Huh7_01', 30),
        7: ('Fluo-C2DL-Huh7_02', 30),

        8: ('Fluo-N2DH-GOWT1_01', 92),
        9: ('Fluo-N2DH-GOWT1_02', 92),

        10: ('Fluo-N2DH-SIM_01', 65),
        11: ('Fluo-N2DH-SIM_02', 150),

        12: ('Fluo-C2DL-MSC_01', 48),
        13: ('Fluo-C2DL-MSC_02', 48),

        14: ('Fluo-N2DL-HeLa_01', 92),
        15: ('Fluo-N2DL-HeLa_02', 92),

        16: ('PhC-C2DH-U373_01', 115),
        17: ('PhC-C2DH-U373_02', 115),
        18: ('PhC-C2DL-PSC_01', 300),
        19: ('PhC-C2DL-PSC_02', 300),

        # test

        20: ('BF-C2DL-HSC_Test_01', 1764),
        21: ('BF-C2DL-HSC_Test_02', 1764),
        22: ('BF-C2DL-MuSC_Test_01', 1376),
        23: ('BF-C2DL-MuSC_Test_02', 1376),

        24: ('DIC-C2DH-HeLa_Test_01', 115),
        25: ('DIC-C2DH-HeLa_Test_02', 115),

        26: ('Fluo-C2DL-Huh7_Test_01', 30),
        27: ('Fluo-C2DL-Huh7_Test_02', 30),

        28: ('Fluo-N2DH-GOWT1_Test_01', 92),
        29: ('Fluo-N2DH-GOWT1_Test_02', 92),

        30: ('Fluo-N2DH-SIM_Test_01', 110),
        31: ('Fluo-N2DH-SIM_Test_02', 138),

        32: ('Fluo-C2DL-MSC_Test_01', 48),
        33: ('Fluo-C2DL-MSC_Test_02', 48),

        34: ('Fluo-N2DL-HeLa_Test_01', 92),
        35: ('Fluo-N2DL-HeLa_Test_02', 92),

        36: ('PhC-C2DH-U373_Test_01', 115),
        37: ('PhC-C2DH-U373_Test_02', 115),
        38: ('PhC-C2DL-PSC_Test_02', 300),
        39: ('PhC-C2DL-PSC_Test_01', 300),
    }


class Params:

    def __init__(self):
        self.db_split = 'huh'

        self.cfg = ()
        self.ignore_missing_gt = 1
        self.ignore_missing_seg = 1
        self.ignored_region_only = 0

        self.resize = 0
        self.root_dir = '/data'
        self.output_dir = '/data'

        self.start_id = 0
        self.end_id = -1
        # self.seq_ids = [6, 7, 14, 15]

        self.write_gt = 0
        self.write_img = 1
        self.raad_gt = 0
        self.tra_only = 0

        self.two_classes = 1

        self.show_img = 0
        self.save_img = 0
        self.save_vid = 0

        self.disable_tqdm = 0
        self.codec = 'H264'
        self.use_tif = 0

        self.vis_height = 1080
        self.vis_width = 1920
        self.db_splits = CTCInfo.DBSplits().__dict__
        self.num_shards = 4


def seg_to_png(gold_seg_src_file_ids, silver_seg_src_file_ids, img_src_file_id,
               silver_seg_path, gold_seg_path, png_seg_src_path, img_src_file, two_classes):
    gold_seg_img = silver_seg_img = None

    try:
        gold_seg_src_file = gold_seg_src_file_ids[img_src_file_id]
    except KeyError:
        n_gold_seg_objs = 0
    else:
        gold_seg_src_path = os.path.join(gold_seg_path, gold_seg_src_file)
        gold_seg_img = cv2.imread(gold_seg_src_path, cv2.IMREAD_UNCHANGED)
        gold_seg_obj_ids = list(np.unique(gold_seg_img, return_counts=False))
        gold_seg_obj_ids.remove(0)

        n_gold_seg_objs = len(gold_seg_obj_ids)

    try:
        silver_seg_src_file = silver_seg_src_file_ids[img_src_file_id]
    except KeyError:
        n_silver_seg_objs = 0
    else:
        silver_seg_src_path = os.path.join(silver_seg_path, silver_seg_src_file)
        silver_seg_img = cv2.imread(silver_seg_src_path, cv2.IMREAD_UNCHANGED)

        silver_seg_obj_ids = list(np.unique(silver_seg_img, return_counts=False))
        silver_seg_obj_ids.remove(0)

        n_silver_seg_objs = len(silver_seg_obj_ids)

    if n_silver_seg_objs == 0 and n_gold_seg_objs == 0:
        # print('\nno segmentations found for {}\n'.format(img_src_file))
        return 0

    if n_silver_seg_objs > n_gold_seg_objs:
        seg_img = silver_seg_img
    else:
        seg_img = gold_seg_img

    if two_classes:
        seg_img[seg_img > 0] = 1
        seg_img = seg_img.astype(np.uint8)

    cv2.imwrite(png_seg_src_path, seg_img)

    return 1


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def _convert_dataset(params):
    """

    :param Params params:
    :return:
    """

    seq_ids = params.db_splits[params.db_split]

    print('root_dir: {}'.format(params.root_dir))
    print('sub_seq: {}'.format(params.db_split))
    print('seq_ids: {}'.format(seq_ids))

    jpg_img_root_path = linux_path(params.root_dir, 'CTC', 'Images')
    tif_img_root_path = linux_path(params.root_dir, 'CTC', 'Images_TIF')
    png_img_root_path = linux_path(params.root_dir, 'CTC', 'Images_PNG')
    os.makedirs(png_img_root_path, exist_ok=True)

    output_root_dir = linux_path(params.root_dir, 'CTC', 'tfrecord')
    os.makedirs(output_root_dir, exist_ok=True)

    tif_labels_root_path = linux_path(params.root_dir, 'CTC', 'tif')
    png_labels_root_path = linux_path(params.root_dir, 'CTC', 'Labels_PNG')
    os.makedirs(png_labels_root_path, exist_ok=True)

    if params.start_id > 0:
        seq_ids = seq_ids[params.start_id:]

    n_seq = len(seq_ids)

    if params.use_tif:
        img_root_path = tif_img_root_path
        img_exts = ('.tif',)
    else:
        img_root_path = jpg_img_root_path
        img_exts = ('.jpg',)

    gold_seg_src_file_ids = {}
    silver_seg_src_file_ids = {}
    img_src_file_ids = {}

    n_total_src_files = 0

    img_src_files = []
    for __id, seq_id in enumerate(seq_ids):

        seq_name, n_frames = CTCInfo.sequences[seq_id]

        print('\tseq {} / {}\t{}\t{}\t{} frames'.format(__id + 1, n_seq, seq_id, seq_name, n_frames))

        silver_seg_path = linux_path(tif_labels_root_path, seq_name + '_ST', 'SEG')
        gold_seg_path = linux_path(tif_labels_root_path, seq_name + '_GT', 'SEG')

        assert os.path.exists(silver_seg_path) or os.path.exists(gold_seg_path), \
            "Neither silver nor gold segmentations found for sequence: {}".format(seq_name)

        if os.path.exists(gold_seg_path):
            _gold_seg_src_files = [linux_path(gold_seg_path, k) for k in os.listdir(gold_seg_path) if
                                   os.path.splitext(k.lower())[1] in ('.tif',)]
            _gold_seg_src_files.sort()

            _gold_seg_src_file_ids = {
                seq_name + '::' + ''.join(k for k in os.path.basename(src_file) if k.isdigit()): src_file
                for src_file in _gold_seg_src_files
            }
            gold_seg_src_file_ids.update(_gold_seg_src_file_ids)
        else:
            print("\ngold  segmentations not found for sequence: {}\n".format(seq_name))

        if os.path.exists(silver_seg_path):
            _silver_seg_src_files = [linux_path(silver_seg_path, k) for k in os.listdir(silver_seg_path) if
                                     os.path.splitext(k.lower())[1] in ('.tif',)]
            _silver_seg_src_files.sort()

            _silver_seg_src_file_ids = {
                seq_name + '::' + ''.join(k for k in os.path.basename(src_file) if k.isdigit()): src_file
                for src_file in _silver_seg_src_files
            }
            silver_seg_src_file_ids.update(_silver_seg_src_file_ids)
        else:
            print("\nsilver  segmentations not found for sequence: {}\n".format(seq_name))

        # unique_gold_seg_src_files = [v for k,v in gold_seg_src_file_ids.items() if k not in
        # silver_seg_src_file_ids.keys()]

        img_dir_path = linux_path(img_root_path, seq_name)
        png_img_dir_path = linux_path(png_img_root_path, seq_name)

        if params.use_tif:
            os.makedirs(png_img_dir_path, exist_ok=True)

        _img_src_files = [linux_path(img_dir_path, k) for k in os.listdir(img_dir_path) if
                          os.path.splitext(k.lower())[1] in img_exts]
        _img_src_files.sort()

        n_total_src_files += len(_img_src_files)

        for img_src_file in _img_src_files:
            img_src_file_no_ext = os.path.splitext(os.path.basename(img_src_file))[0]
            img_src_file_id = seq_name + '::' + ''.join(k for k in os.path.basename(img_src_file) if k.isdigit())

            silver_seg_path = linux_path(tif_labels_root_path, seq_name + '_ST', 'SEG')
            gold_seg_path = linux_path(tif_labels_root_path, seq_name + '_GT', 'SEG')
            png_seg_path = linux_path(png_labels_root_path, seq_name)
            os.makedirs(png_seg_path, exist_ok=True)

            png_seg_src_path = os.path.join(png_seg_path, img_src_file_no_ext + '.png')
            if not os.path.exists(png_seg_src_path):
                segmentation_found = seg_to_png(gold_seg_src_file_ids, silver_seg_src_file_ids, img_src_file_id,
                                                silver_seg_path, gold_seg_path, png_seg_src_path, img_src_file,
                                                params.two_classes)
                if not segmentation_found:
                    continue

            png_img_dir_path = linux_path(png_img_root_path, seq_name)
            jpg_img_dir_path = linux_path(jpg_img_root_path, seq_name)
            tif_img_dir_path = linux_path(tif_img_root_path, seq_name)

            if params.use_tif:
                png_img_src_path = os.path.join(png_img_dir_path, img_src_file)
                if not os.path.exists(png_img_src_path):
                    tif_img_src_path = os.path.join(tif_img_dir_path, img_src_file)
                    img = cv2.imread(tif_img_src_path, cv2.IMREAD_UNCHANGED)
                    cv2.imwrite(png_img_src_path, img)
                img_src_path = png_img_src_path
            else:
                jpg_img_src_path = os.path.join(jpg_img_dir_path, img_src_file)
                img_src_path = jpg_img_src_path

            img_src_files.append(img_src_file)
            img_src_file_ids[img_src_file] = (img_src_file_id, seq_name, png_seg_src_path, img_src_path)

    n_src_files = len(img_src_files)
    print('\n\n{}: {} / {}\n\n'.format(params.db_split, n_src_files, n_total_src_files))

    # return

    # output_dir = linux_path(output_root_dir, seq_name)
    output_dir = output_root_dir

    create_tfrecords(img_src_files, img_src_file_ids, params.num_shards, params.db_split, params.use_tif, output_dir)


def create_tfrecords(src_files, file_ids, n_shards, sub_seq, use_tif, output_dir):
    if use_tif:
        image_reader = build_data.ImageReader('png', channels=1)
    else:
        image_reader = build_data.ImageReader('jpeg', channels=1)

    label_reader = build_data.ImageReader('png', channels=1)

    n_images = len(src_files)
    n_per_shard = int(math.ceil(n_images / float(n_shards)))

    os.makedirs(output_dir, exist_ok=True)

    print('Creating {} shards with {} images ({} per shard)'.format(n_shards, n_images, n_per_shard))

    for shard_id in range(n_shards):

        output_file_path = os.path.join(
            output_dir,
            '{:s}-{:05d}-of-{:05d}.tfrecord'.format(sub_seq, shard_id, n_shards))

        with tf.python_io.TFRecordWriter(output_file_path) as tfrecord_writer:
            start_idx = shard_id * n_per_shard
            end_idx = min((shard_id + 1) * n_per_shard, n_images)

            for img_id in tqdm(range(start_idx, end_idx), ncols=50):

                img_src_file = src_files[img_id]
                img_src_file_id, seq_name, seg_src_path, img_src_path = file_ids[img_src_file]

                seg_data = tf.gfile.FastGFile(seg_src_path, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)

                image_data = tf.gfile.FastGFile(img_src_path, 'rb').read()

                height, width = image_reader.read_image_dims(image_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatch found between image and label')

                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, img_src_path, height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())


def main():
    params = Params()
    paramparse.process(params)

    # for _sub_seq in params.sub_seq_dict:
    #     params.sub_seq = _sub_seq
    #     _convert_dataset(params)

    _convert_dataset(params)


if __name__ == '__main__':
    main()
