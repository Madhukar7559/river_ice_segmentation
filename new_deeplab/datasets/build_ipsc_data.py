"""Converts IPSC data to TFRecord file format with Example protos."""

import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import sys
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import paramparse

from ipsc_info import IPSCInfo
import build_data


def irange(a, b):
    return list(range(a, b + 1))


class Params:

    def __init__(self):
        self.db_split = 'all'

        self.cfg = ()
        self.ignore_missing_gt = 1
        self.ignore_missing_seg = 1
        self.ignored_region_only = 0

        self.resize = 0
        self.root_dir = '/data/ipsc/201020/masks'
        self.output_dir = '/data/tfrecord/ipsc'

        self.start_id = 0
        self.end_id = -1

        self.write_gt = 0
        self.write_img = 1
        self.raad_gt = 0
        self.tra_only = 0

        self.two_classes = 1

        self.show_img = 0
        self.save_img = 0
        self.save_vid = 0

        self.disable_seg = 0
        self.disable_tqdm = 0
        self.codec = 'H264'

        self.vis_height = 1080
        self.vis_width = 1920
        self.num_shards = 4

        self.db_splits = IPSCInfo.DBSplits().__dict__


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def create_tfrecords(img_src_files, seg_src_files, n_shards, db_split, output_dir):
    image_reader = build_data.ImageReader('jpeg', channels=1)

    label_reader = build_data.ImageReader('png', channels=1)

    n_images = len(img_src_files)
    n_per_shard = int(math.ceil(n_images / float(n_shards)))

    os.makedirs(output_dir, exist_ok=True)

    print('Creating {} shards with {} images ({} per shard)'.format(n_shards, n_images, n_per_shard))

    for shard_id in range(n_shards):

        output_file_path = os.path.join(
            output_dir,
            '{:s}-{:05d}-of-{:05d}.tfrecord'.format(db_split, shard_id, n_shards))

        with tf.python_io.TFRecordWriter(output_file_path) as tfrecord_writer:
            start_idx = shard_id * n_per_shard
            end_idx = min((shard_id + 1) * n_per_shard, n_images)

            for img_id in tqdm(range(start_idx, end_idx), ncols=50):

                img_src_path = img_src_files[img_id]

                image_data = tf.gfile.FastGFile(img_src_path, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)

                if seg_src_files is not None:
                    seg_src_path = seg_src_files[img_id]
                    seg_data = tf.gfile.FastGFile(seg_src_path, 'rb').read()
                    seg_height, seg_width = label_reader.read_image_dims(seg_data)
                    assert height == seg_height and width == seg_width, 'Shape mismatch found between image and label'
                else:
                    seg_data = None

                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, img_src_path, height, width, seg_data)

                tfrecord_writer.write(example.SerializeToString())


def _convert_dataset(params):
    """

    :param Params params:
    :return:
    """

    seq_ids = params.db_splits[params.db_split]

    print('root_dir: {}'.format(params.root_dir))
    print('db_split: {}'.format(params.db_split))
    print('seq_ids: {}'.format(seq_ids))
    print('output_dir: {}'.format(params.output_dir))

    os.makedirs(params.output_dir, exist_ok=True)

    if params.disable_seg:
        print('\nSegmentations are disabled\n')

    if params.start_id > 0:
        seq_ids = seq_ids[params.start_id:]

    n_seq = len(seq_ids)
    img_exts = ('.jpg',)

    n_total_src_files = 0

    img_src_files = []
    seg_src_files = []
    for __id, seq_id in enumerate(seq_ids):

        seq_name, n_frames = IPSCInfo.sequences[seq_id]

        print('\tseq {} / {}\t{}\t{}\t{} frames'.format(__id + 1, n_seq, seq_id, seq_name, n_frames))

        img_dir_path = linux_path(params.root_dir, seq_name, 'images')

        _img_src_files = [linux_path(img_dir_path, k) for k in os.listdir(img_dir_path) if
                          os.path.splitext(k.lower())[1] in img_exts]
        _img_src_files.sort()

        img_src_files += _img_src_files

        n_img_src_files = len(_img_src_files)

        n_total_src_files += len(_img_src_files)

        assert n_img_src_files == n_frames, \
            "Mismatch between number of the specified frames and number of actual images in folder"

        if not params.disable_seg:
            seg_path = linux_path(params.root_dir, seq_name, 'labels')

            assert os.path.exists(seg_path), \
                "segmentations found for sequence: {}".format(seq_name)

            _seg_src_files = [linux_path(seg_path, k) for k in os.listdir(seg_path) if
                              os.path.splitext(k.lower())[1] in ('.png',)]
            _seg_src_files.sort()
            n_seg_src_files = len(_seg_src_files)

            assert n_img_src_files == n_seg_src_files, "mismatch between number of source and segmentation images"
            seg_src_files += _seg_src_files

    create_tfrecords(img_src_files, seg_src_files, params.num_shards, params.db_split, params.output_dir)


def main():
    params = Params()
    paramparse.process(params)

    _convert_dataset(params)


if __name__ == '__main__':
    main()
