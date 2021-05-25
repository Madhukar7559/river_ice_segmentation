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


# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string(
#     'train_image_folder',
#     './ADE20K/ADEChallengeData2016/images/training',
#     'Folder containing trainng images')
# tf.app.flags.DEFINE_string(
#     'train_image_label_folder',
#     './ADE20K/ADEChallengeData2016/annotations/training',
#     'Folder containing annotations for trainng images')
#
# tf.app.flags.DEFINE_string(
#     'val_image_folder',
#     './ADE20K/ADEChallengeData2016/images/validation',
#     'Folder containing validation images')
#
# tf.app.flags.DEFINE_string(
#     'val_image_label_folder',
#     './ADE20K/ADEChallengeData2016/annotations/validation',
#     'Folder containing annotations for validation')
#
# tf.app.flags.DEFINE_string(
#     'output_dir', './ADE20K/tfrecord',
#     'Path to save converted tfrecord of Tensorflow example')


def irange(a, b):
    return list(range(a, b + 1))


class Params:
    class CTCSubSeq:
        def __init__(self):
            self.all = irange(0, 19)
            self.bf = irange(0, 3)
            self.bf1 = irange(0, 1)
            self.bf2 = irange(2, 3)
            self.dic = irange(4, 5)
            self.fluo = irange(6, 15)
            self.fluo1 = irange(6, 11)
            self.fluo2 = irange(12, 15)
            self.huh = irange(6, 7)
            self.gow = irange(8, 9)
            self.sim = irange(10, 11)
            self.hela = irange(14, 15)
            self.phc = irange(16, 19)
            self.phc1 = irange(16, 17)
            self.phc2 = irange(18, 19)

    def __init__(self):
        self.cfg = ('',)
        self.ignore_missing_gt = 1
        self.ignore_missing_seg = 1
        self.ignored_region_only = 0

        self.resize = 0
        self.root_dir = '/data'
        self.output_dir = '/data'

        self.start_id = 0
        self.end_id = -1
        self.sub_seq = 'bf'
        # self.seq_ids = [6, 7, 14, 15]

        self.write_gt = 0
        self.write_img = 1
        self.raad_gt = 0
        self.tra_only = 0

        self.show_img = 0
        self.save_img = 0
        self.save_vid = 0

        self.disable_tqdm = 0
        self.codec = 'H264'
        self.use_tif = 0

        self.vis_height = 1080
        self.vis_width = 1920
        self.sub_seq_dict = Params.CTCSubSeq().__dict__
        self.num_shards = 4

        self.sequences = {
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


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def _convert_dataset(params):
    """

    :param Params params:
    :return:
    """

    seq_ids = params.sub_seq_dict[params.sub_seq]

    print('root_dir: {}'.format(params.root_dir))
    print('sub_seq: {}'.format(params.sub_seq))
    print('seq_ids: {}'.format(seq_ids))

    jpg_img_root_path = linux_path(params.root_dir, 'CTC', 'Images')
    tif_img_root_path = linux_path(params.root_dir, 'CTC', 'Images_TIF')
    png_img_root_path = linux_path(params.root_dir, 'CTC', 'Images_PNG')
    os.makedirs(png_img_root_path, exist_ok=True)

    output_root_dir = linux_path(params.root_dir, 'CTC', 'tfrecord', params.sub_seq)
    os.makedirs(output_root_dir, exist_ok=True)

    tif_labels_root_path = linux_path(params.root_dir, 'CTC', 'tif')
    png_labels_root_path = linux_path(params.root_dir, 'CTC', 'Labels_PNG')
    os.makedirs(png_labels_root_path, exist_ok=True)

    for seq_id in seq_ids:

        seq_name = params.sequences[seq_id][0]

        silver_seg_path = linux_path(tif_labels_root_path, seq_name + '_ST', 'SEG')
        gold_seg_path = linux_path(tif_labels_root_path, seq_name + '_GT', 'SEG')

        png_seg_path = linux_path(png_labels_root_path, seq_name)
        os.makedirs(png_seg_path, exist_ok=True)

        assert os.path.exists(silver_seg_path) or os.path.exists(gold_seg_path), \
            "Neither silver nor gold segmentations found for sequence: {}".format(seq_name)

        gold_seg_src_file_ids = {}
        silver_seg_src_file_ids = {}

        if os.path.exists(gold_seg_path):
            gold_seg_src_files = [k for k in os.listdir(gold_seg_path) if
                                  os.path.splitext(k.lower())[1] in ('.tif',)]
            gold_seg_src_files.sort()

            gold_seg_src_file_ids = {''.join(k for k in src_file if k.isdigit()): src_file
                                     for src_file in gold_seg_src_files}
        else:
            print("\ngold  segmentations not found for sequence: {}\n".format(seq_name))

        if os.path.exists(silver_seg_path):
            silver_seg_src_files = [k for k in os.listdir(silver_seg_path) if
                                    os.path.splitext(k.lower())[1] in ('.tif',)]
            silver_seg_src_files.sort()

            silver_seg_src_file_ids = {''.join(k for k in src_file if k.isdigit()): src_file
                                       for src_file in silver_seg_src_files}
        else:
            print("\nsilver  segmentations not found for sequence: {}\n".format(seq_name))

        # unique_gold_seg_src_files = [v for k,v in gold_seg_src_file_ids.items() if k not in
        # silver_seg_src_file_ids.keys()]

        tif_img_dir_path = linux_path(tif_img_root_path, seq_name)
        png_img_dir_path = linux_path(png_img_root_path, seq_name)
        jpg_img_dir_path = linux_path(jpg_img_root_path, seq_name)

        if params.use_tif:
            img_dir_path = tif_img_dir_path
            os.makedirs(png_img_dir_path, exist_ok=True)
            img_exts = ('.tif',)
            image_reader = build_data.ImageReader('png', channels=1)
        else:
            img_dir_path = jpg_img_dir_path
            img_exts = ('.jpg',)
            image_reader = build_data.ImageReader('jpeg', channels=1)

        img_src_files = [k for k in os.listdir(img_dir_path) if
                         os.path.splitext(k.lower())[1] in img_exts]
        img_src_files.sort()
        label_reader = build_data.ImageReader('png', channels=1)

        num_images = len(img_src_files)
        num_per_shard = int(math.ceil(num_images / float(params.num_shards)))

        # output_dir = linux_path(output_root_dir, seq_name)
        output_dir = output_root_dir
        os.makedirs(output_dir, exist_ok=True)

        for shard_id in range(params.num_shards):

            output_file_path = os.path.join(
                output_dir,
                '{:s}-{:05d}-of-{:05d}.tfrecord'.format(seq_name, shard_id, params.num_shards))

            with tf.python_io.TFRecordWriter(output_file_path) as tfrecord_writer:
                start_idx = shard_id * num_per_shard
                end_idx = min((shard_id + 1) * num_per_shard, num_images)

                for img_id in range(start_idx, end_idx):

                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                        img_id + 1, num_images, shard_id))

                    img_src_file = img_src_files[img_id]
                    img_src_file_no_ext = os.path.splitext(os.path.basename(img_src_file))[0]
                    img_src_file_id = ''.join(k for k in img_src_file if k.isdigit())

                    png_seg_src_path = os.path.join(png_seg_path, img_src_file_no_ext + '.png')
                    if not os.path.exists(png_seg_src_path):
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
                            print('\nno segmentations found for {}\n'.format(img_src_file))
                            continue

                        if n_silver_seg_objs > n_gold_seg_objs:
                            cv2.imwrite(png_seg_src_path, silver_seg_img)
                        else:
                            cv2.imwrite(png_seg_src_path, gold_seg_img)

                    seg_data = tf.gfile.FastGFile(png_seg_src_path, 'rb').read()
                    seg_height, seg_width = label_reader.read_image_dims(seg_data)

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

                    image_data = tf.gfile.FastGFile(img_src_path, 'rb').read()

                    height, width = image_reader.read_image_dims(image_data)
                    if height != seg_height or width != seg_width:
                        raise RuntimeError('Shape mismatched between image and label.')
                    # Convert to tf example.
                    example = build_data.image_seg_to_tfexample(
                        image_data, jpg_img_src_path, height, width, seg_data)
                    tfrecord_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()


def main():
    params = Params()
    paramparse.process(params)
    _convert_dataset(params)


if __name__ == '__main__':
    main()
