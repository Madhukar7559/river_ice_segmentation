import os, sys
import numpy as np
import imageio

from paramparse import MultiPath

import densenet.evaluation.eval_segm as eval
from densenet.utils import read_data, getDateTime, print_and_write, linux_path
from datasets.build_utils import convert_to_raw_mask

import cv2


class VisParams:

    def __init__(self):
        self.cfg = ()
        self.end_id = -1
        self.images_ext = 'png'
        self.labels_ext = 'png'
        self.log_dir = ''
        self.n_classes = 3
        self.normalize_labels = 1
        self.out_ext = 'png'
        self.save_path = ''
        self.save_stitched = 0
        self.seg_ext = 'png'
        self.seg_path = ''
        self.selective_mode = 0
        self.show_img = 0
        self.start_id = 0
        self.stitch = 0
        self.stitch_seg = 1
        self.no_labels = 1

        self.multi_sequence_db = 0
        self.seg_on_subset = 0

        self.log_root_dir = 'log'
        self.db_root_dir = '/data'

        self.images_path = ''
        self.labels_path = ''
        self.labels_dir = 'labels'
        self.images_dir = 'images'

        self.dataset = ''

        self.model_info = MultiPath()
        self.train_split = MultiPath()
        self.vis_split = MultiPath()
        self.train_info = MultiPath()
        self.vis_info = MultiPath()

    def process(self):
        if not self.images_path:
            self.images_path = os.path.join(self.db_root_dir, self.dataset, self.images_dir)

        if not self.labels_path:
            self.labels_path = os.path.join(self.db_root_dir, self.dataset, self.labels_dir)

        log_dir = linux_path(self.log_root_dir, self.train_info, self.model_info)

        if not self.seg_path:
            self.seg_path = linux_path(log_dir, self.vis_info, 'raw')

        if not self.save_path:
            self.save_path = linux_path(log_dir, self.vis_info, 'vis')


def run(params):
    """

    :param VisParams params:
    :return:
    """

    if params.multi_sequence_db:
        assert params.vis_split, "vis_split must be provided for CTC"

        """some repeated code here to allow better IntelliSense"""
        if params.dataset.lower() == 'ctc':
            from new_deeplab.datasets.build_ctc_data import CTCInfo
            db_splits = CTCInfo.DBSplits().__dict__
            sequences = CTCInfo.sequences
        elif params.dataset.lower() == 'ipsc':
            from new_deeplab.datasets.ipsc_info import IPSCInfo
            db_splits = IPSCInfo.DBSplits().__dict__
            sequences = IPSCInfo.sequences
        elif params.dataset.lower() == 'ipsc_patches':
            from new_deeplab.datasets.ipsc_info import IPSCPatchesInfo
            db_splits = IPSCPatchesInfo.DBSplits().__dict__
            sequences = IPSCPatchesInfo.sequences
        else:
            raise AssertionError('multi_sequence_db {} is not supported yet'.format(params.dataset))

        seq_ids = db_splits[params.vis_split]

        src_files = []
        seg_labels_list = []
        if params.no_labels:
            src_labels_list = None
        else:
            src_labels_list = []

        total_frames = 0
        seg_total_frames = 0

        for seq_id in seq_ids:
            seq_name, n_frames = sequences[seq_id]

            images_path = os.path.join(params.images_path, seq_name)

            if params.no_labels:
                labels_path = ''
            else:
                labels_path = os.path.join(params.labels_path, seq_name)

            _src_files, _src_labels_list, _total_frames = read_data(images_path, params.images_ext,
                                                                    labels_path,
                                                                    params.labels_ext)

            _src_filenames = [os.path.splitext(os.path.basename(k))[0] for k in _src_files]

            if not params.no_labels:
                _src_labels_filenames = [os.path.splitext(os.path.basename(k))[0] for k in _src_labels_list]

                assert _src_labels_filenames == _src_filenames, "mismatch between image and label filenames"

            eval_mode = False
            if params.seg_path and params.seg_ext:
                seg_path = os.path.join(params.seg_path, seq_name)

                _, _seg_labels_list, _seg_total_frames = read_data(labels_path=seg_path, labels_ext=params.seg_ext,
                                                                   labels_type='seg')

                _seg_labels__filenames = [os.path.splitext(os.path.basename(k))[0] for k in _seg_labels_list]

                if _seg_total_frames != _total_frames:

                    if params.seg_on_subset and _seg_total_frames < _total_frames:
                        matching_ids = [_src_filenames.index(k) for k in _seg_labels__filenames]

                        _src_files = [_src_files[i] for i in matching_ids]
                        if not params.no_labels:
                            _src_labels_list = [_src_labels_list[i] for i in matching_ids]

                        _total_frames = _seg_total_frames

                    else:
                        raise AssertionError('Mismatch between no. of frames in GT and seg labels: {} and {}'.format(
                            _total_frames, _seg_total_frames))

                    seg_labels_list += _seg_labels_list

                seg_total_frames += _seg_total_frames
                eval_mode = True

            src_files += _src_files
            if not params.no_labels:
                src_labels_list += _src_labels_list
            else:
                params.stitch = params.save_stitched = 1

            total_frames += _total_frames
    else:
        src_files, src_labels_list, total_frames = read_data(params.images_path, params.images_ext, params.labels_path,
                                                             params.labels_ext)

        eval_mode = False
        if params.labels_path and params.seg_path and params.seg_ext:
            _, seg_labels_list, seg_total_frames = read_data(labels_path=params.seg_path, labels_ext=params.seg_ext,
                                                             labels_type='seg')
            if seg_total_frames != total_frames:
                raise SystemError('Mismatch between no. of frames in GT and seg labels: {} and {}'.format(
                    total_frames, seg_total_frames))
            eval_mode = True
        else:
            params.stitch = params.save_stitched = 1

    if params.end_id < params.start_id:
        params.end_id = total_frames - 1

    if not params.save_path:
        if eval_mode:
            params.save_path = os.path.join(os.path.dirname(params.seg_path), 'vis')
        else:
            params.save_path = os.path.join(os.path.dirname(params.images_path), 'vis')

    if not os.path.isdir(params.save_path):
        os.makedirs(params.save_path)
    if params.stitch and params.save_stitched:
        print('Saving visualization images to: {}'.format(params.save_path))

    log_fname = os.path.join(params.save_path, 'vis_log_{:s}.txt'.format(getDateTime()))
    print('Saving log to: {}'.format(log_fname))

    save_path_parent = os.path.dirname(params.save_path)
    templ_1 = os.path.basename(save_path_parent)
    templ_2 = os.path.basename(os.path.dirname(save_path_parent))

    templ = '{}_{}'.format(templ_1, templ_2)

    if params.selective_mode:
        label_diff = int(255.0 / params.n_classes)
    else:
        label_diff = int(255.0 / (params.n_classes - 1))

    print('templ: {}'.format(templ))
    print('label_diff: {}'.format(label_diff))

    n_frames = params.end_id - params.start_id + 1

    pix_acc = np.zeros((n_frames,))

    mean_acc = np.zeros((n_frames,))
    # mean_acc_ice = np.zeros((n_frames,))
    # mean_acc_ice_1 = np.zeros((n_frames,))
    # mean_acc_ice_2 = np.zeros((n_frames,))
    #
    mean_IU = np.zeros((n_frames,))
    # mean_IU_ice = np.zeros((n_frames,))
    # mean_IU_ice_1 = np.zeros((n_frames,))
    # mean_IU_ice_2 = np.zeros((n_frames,))

    fw_IU = np.zeros((n_frames,))
    fw_sum = np.zeros((params.n_classes,))

    print_diff = max(1, int(n_frames * 0.01))

    avg_mean_acc_ice = avg_mean_acc_ice_1 = avg_mean_acc_ice_2 = 0
    avg_mean_IU_ice = avg_mean_IU_ice_1 = avg_mean_IU_ice_2 = 0

    skip_mean_acc_ice_1 = skip_mean_acc_ice_2 = 0
    skip_mean_IU_ice_1 = skip_mean_IU_ice_2 = 0
    _pause = 1
    labels_img = None

    for img_id in range(params.start_id, params.end_id + 1):

        # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
        src_img_fname = src_files[img_id]
        img_fname = os.path.basename(src_img_fname)

        img_fname_no_ext = os.path.splitext(img_fname)[0]

        if params.stitch or params.show_img:
            # src_img_fname = os.path.join(params.images_path, img_fname)
            src_img = imageio.imread(src_img_fname)
            if src_img is None:
                raise SystemError('Source image could not be read from: {}'.format(src_img_fname))

            try:
                src_height, src_width, _ = src_img.shape
            except ValueError as e:
                print('src_img_fname: {}'.format(src_img_fname))
                print('src_img: {}'.format(src_img))
                print('src_img.shape: {}'.format(src_img.shape))
                print('error: {}'.format(e))
                sys.exit(1)

        if not params.labels_path:
            stitched = src_img
        else:
            # labels_img_fname = os.path.join(params.labels_path, img_fname_no_ext + '.{}'.format(params.labels_ext))
            labels_img_fname = src_labels_list[img_id]

            labels_img_orig = imageio.imread(labels_img_fname)
            if labels_img_orig is None:
                raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))

            _, src_width = labels_img_orig.shape[:2]

            if len(labels_img_orig.shape) == 3:
                labels_img_orig = np.squeeze(labels_img_orig[:, :, 0])

            if params.show_img:
                cv2.imshow('labels_img_orig', labels_img_orig)

            labels_img_orig = convert_to_raw_mask(labels_img_orig, params.n_classes, labels_img_fname)

            labels_img = np.copy(labels_img_orig)
            # if params.normalize_labels:
            #     if params.selective_mode:
            #         selective_idx = (labels_img_orig == 255)
            #         print('labels_img_orig.shape: {}'.format(labels_img_orig.shape))
            #         print('selective_idx count: {}'.format(np.count_nonzero(selective_idx)))
            #         labels_img_orig[selective_idx] = params.n_classes
            #         if params.show_img:
            #             cv2.imshow('labels_img_orig norm', labels_img_orig)
            #     labels_img = (labels_img_orig.astype(np.float64) * label_diff).astype(np.uint8)
            # else:
            #     labels_img = np.copy(labels_img_orig)

            if len(labels_img.shape) != 3:
                labels_img = np.stack((labels_img, labels_img, labels_img), axis=2)

            if params.stitch:
                stitched = np.concatenate((src_img, labels_img), axis=1)

            if eval_mode:
                # seg_img_fname = os.path.join(params.seg_path, img_fname_no_ext + '.{}'.format(params.seg_ext))
                seg_img_fname = seg_labels_list[img_id]

                seg_img = imageio.imread(seg_img_fname)
                if seg_img is None:
                    raise SystemError('Segmentation image could not be read from: {}'.format(seg_img_fname))

                # seg_img = convert_to_raw_mask(seg_img, params.n_classes, seg_img_fname)

                if len(seg_img.shape) == 3:
                    seg_img = np.squeeze(seg_img[:, :, 0])

                eval_cl, _ = eval.extract_classes(seg_img)
                gt_cl, _ = eval.extract_classes(labels_img_orig)

                if seg_img.max() > params.n_classes - 1:
                    seg_img = (seg_img.astype(np.float64) / label_diff).astype(np.uint8)

                seg_height, seg_width = seg_img.shape

                if seg_width == 2 * src_width or seg_width == 3 * src_width:
                    _start_id = seg_width - src_width
                    seg_img = seg_img[:, _start_id:]

                # print('seg_img.shape: ', seg_img.shape)
                # print('labels_img_orig.shape: ', labels_img_orig.shape)

                pix_acc[img_id] = eval.pixel_accuracy(seg_img, labels_img_orig)
                _acc, mean_acc[img_id] = eval.mean_accuracy(seg_img, labels_img_orig, return_acc=1)
                _IU, mean_IU[img_id] = eval.mean_IU(seg_img, labels_img_orig, return_iu=1)
                fw_IU[img_id], _fw = eval.frequency_weighted_IU(seg_img, labels_img_orig, return_freq=1)
                try:
                    fw_sum += _fw
                except ValueError as e:
                    print('fw_sum: {}'.format(fw_sum))
                    print('_fw: {}'.format(_fw))

                    eval_cl, _ = eval.extract_classes(seg_img)
                    gt_cl, _ = eval.extract_classes(labels_img_orig)
                    cl = np.union1d(eval_cl, gt_cl)

                    print('cl: {}'.format(cl))
                    print('eval_cl: {}'.format(eval_cl))
                    print('gt_cl: {}'.format(gt_cl))

                    raise ValueError(e)
                mean_acc_ice = np.mean(list(_acc.values())[1:])
                avg_mean_acc_ice += (mean_acc_ice - avg_mean_acc_ice) / (img_id + 1)
                try:
                    mean_acc_ice_1 = _acc[1]
                    avg_mean_acc_ice_1 += (mean_acc_ice_1 - avg_mean_acc_ice_1) / (img_id - skip_mean_acc_ice_1 + 1)
                except KeyError:
                    print('\nskip_mean_acc_ice_1: {}'.format(img_id))
                    skip_mean_acc_ice_1 += 1
                try:
                    mean_acc_ice_2 = _acc[2]
                    avg_mean_acc_ice_2 += (mean_acc_ice_2 - avg_mean_acc_ice_2) / (img_id - skip_mean_acc_ice_2 + 1)
                except KeyError:
                    print('\nskip_mean_acc_ice_2: {}'.format(img_id))
                    skip_mean_acc_ice_2 += 1

                mean_IU_ice = np.mean(list(_IU.values())[1:])
                avg_mean_IU_ice += (mean_IU_ice - avg_mean_IU_ice) / (img_id + 1)
                try:
                    mean_IU_ice_1 = _IU[1]
                    avg_mean_IU_ice_1 += (mean_IU_ice_1 - avg_mean_IU_ice_1) / (img_id - skip_mean_IU_ice_1 + 1)
                except KeyError:
                    print('\nskip_mean_IU_ice_1: {}'.format(img_id))
                    skip_mean_IU_ice_1 += 1
                try:
                    mean_IU_ice_2 = _IU[2]
                    avg_mean_IU_ice_2 += (mean_IU_ice_2 - avg_mean_IU_ice_2) / (img_id - skip_mean_IU_ice_2 + 1)
                except KeyError:
                    print('\nskip_mean_IU_ice_2: {}'.format(img_id))
                    skip_mean_IU_ice_2 += 1

                seg_img = (seg_img * label_diff).astype(np.uint8)
                if len(seg_img.shape) != 3:
                    seg_img = np.stack((seg_img, seg_img, seg_img), axis=2)

                if params.stitch and params.stitch_seg:
                    stitched = np.concatenate((stitched, seg_img), axis=1)
                if not params.stitch and params.show_img:
                    cv2.imshow('seg_img', seg_img)
            else:
                _, _fw = eval.frequency_weighted_IU(labels_img_orig, labels_img_orig, return_freq=1)
                try:
                    fw_sum += _fw
                except ValueError as e:
                    print('fw_sum: {}'.format(fw_sum))
                    print('_fw: {}'.format(_fw))

                    gt_cl, _ = eval.extract_classes(labels_img_orig)
                    print('gt_cl: {}'.format(gt_cl))
                    for k in range(params.n_classes):
                        if k not in gt_cl:
                            _fw.insert(k, 0)

                    fw_sum += _fw

            _fw_total = np.sum(_fw)

            # print('_fw: {}'.format(_fw))
            # print('_fw_total: {}'.format(_fw_total))

            _fw_frac = np.array(_fw) / float(_fw_total)

            print('_fw_frac: {}'.format(_fw_frac))

        if params.stitch:
            if params.save_stitched:
                seg_save_path = os.path.join(params.save_path, '{}.{}'.format(img_fname_no_ext, params.out_ext))
                imageio.imsave(seg_save_path, stitched)
            if params.show_img:
                cv2.imshow('stitched', stitched)
        else:
            if params.show_img:
                cv2.imshow('src_img', src_img)
                if params.labels_path:
                    cv2.imshow('labels_img', labels_img)

        if params.show_img:
            k = cv2.waitKey(1 - _pause)
            if k == 27:
                sys.exit(0)
            elif k == 32:
                _pause = 1 - _pause
        img_done = img_id - params.start_id + 1
        if img_done % print_diff == 0:
            log_txt = 'Done {:5d}/{:5d} frames'.format(img_done, n_frames)
            if eval_mode:
                log_txt = '{:s} pix_acc: {:.5f} mean_acc: {:.5f} mean_IU: {:.5f} fw_IU: {:.5f}' \
                          ' acc_ice: {:.5f} acc_ice_1: {:.5f} acc_ice_2: {:.5f}' \
                          ' IU_ice: {:.5f} IU_ice_1: {:.5f} IU_ice_2: {:.5f}'.format(
                    log_txt, pix_acc[img_id], mean_acc[img_id], mean_IU[img_id], fw_IU[img_id],
                    avg_mean_acc_ice, avg_mean_acc_ice_1, avg_mean_acc_ice_2,
                    avg_mean_IU_ice, avg_mean_IU_ice_1, avg_mean_IU_ice_2,
                )
            print_and_write(log_txt, log_fname)

    sys.stdout.write('\n')
    sys.stdout.flush()

    if eval_mode:
        log_txt = "pix_acc\t mean_acc\t mean_IU\t fw_IU\n{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
            np.mean(pix_acc), np.mean(mean_acc), np.mean(mean_IU), np.mean(fw_IU))
        log_txt += "mean_acc_ice\t mean_acc_ice_1\t mean_acc_ice_2\n{:.5f}\t{:.5f}\t{:.5f}\n".format(
            avg_mean_acc_ice, avg_mean_acc_ice_1, avg_mean_acc_ice_2)
        log_txt += "mean_IU_ice\t mean_IU_ice_1\t mean_IU_ice_2\n{:.5f}\t{:.5f}\t{:.5f}\n".format(
            avg_mean_IU_ice, avg_mean_IU_ice_1, avg_mean_IU_ice_2)
        print_and_write(log_txt, log_fname)

        log_txt = "{}\n\tanchor_ice\t frazil_ice\t ice+water\t ice+water(fw)\n" \
                  "recall\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n" \
                  "precision\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(templ,
                                                                       avg_mean_acc_ice_1, avg_mean_acc_ice_2,
                                                                       np.mean(mean_acc), np.mean(pix_acc),
                                                                       avg_mean_IU_ice_1, avg_mean_IU_ice_2,
                                                                       np.mean(mean_IU), np.mean(fw_IU),
                                                                       )
        print_and_write(log_txt, log_fname)

    fw_sum_total = np.sum(fw_sum)
    fw_sum_frac = fw_sum / float(fw_sum_total)

    print('fw_sum_total: {}'.format(fw_sum_total))
    print('fw_sum_frac: {}'.format(fw_sum_frac))

    print('Wrote log to: {}'.format(log_fname))


if __name__ == '__main__':
    import paramparse

    _params = VisParams()
    paramparse.process(_params)
    _params.process()
    run(_params)
