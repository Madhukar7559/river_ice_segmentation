import argparse, os, sys
import numpy as np
from scipy.misc.pilutil import imread, imsave
import densenet.evaluation.eval_segm as eval
from densenet.utils import readData, getDateTime, print_and_write

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str)

parser.add_argument("--images_path", type=str)
parser.add_argument("--images_ext", type=str, default='png')
parser.add_argument("--labels_path", type=str, default='')
parser.add_argument("--labels_ext", type=str, default='png')
parser.add_argument("--seg_path", type=str, default='')
parser.add_argument("--seg_ext", type=str, default='png')

parser.add_argument("--out_ext", type=str, default='png')

parser.add_argument("--save_path", type=str, default='')

parser.add_argument("--n_classes", type=int)

parser.add_argument("--save_stitched", type=int, default=0)

parser.add_argument("--start_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=-1)

parser.add_argument("--show_img", type=int, default=0)
parser.add_argument("--stitch", type=int, default=0)
parser.add_argument("--stitch_seg", type=int, default=1)

parser.add_argument("--normalize_labels", type=int, default=1)
parser.add_argument("--selective_mode", type=int, default=0)

args = parser.parse_args()

images_path = args.images_path
images_ext = args.images_ext
labels_path = args.labels_path
labels_ext = args.labels_ext

seg_path = args.seg_path
seg_ext = args.seg_ext

out_ext = args.out_ext

save_path = args.save_path

n_classes = args.n_classes

end_id = args.end_id
start_id = args.start_id

show_img = args.show_img
stitch = args.stitch
stitch_seg = args.stitch_seg
save_stitched = args.save_stitched

normalize_labels = args.normalize_labels
selective_mode = args.selective_mode

src_files, src_labels_list, total_frames = readData(images_path, images_ext, labels_path,
                                                    labels_ext)
if end_id < start_id:
    end_id = total_frames - 1

eval_mode = False
if labels_path and seg_path and seg_ext:
    _, seg_labels_list, seg_total_frames = readData(labels_path=seg_path, labels_ext=seg_ext, labels_type='seg')
    if seg_total_frames != total_frames:
        raise SystemError('Mismatch between no. of frames in GT and seg labels: {} and {}'.format(
            total_frames, seg_total_frames))
    eval_mode = True
else:
    stitch = save_stitched = 1

if not save_path:
    if eval_mode:
        save_path = os.path.join(os.path.dirname(seg_path), 'vis')
    else:
        save_path = os.path.join(os.path.dirname(images_path), 'vis')

if not os.path.isdir(save_path):
    os.makedirs(save_path)
if stitch and save_stitched:
    print('Saving visualization images to: {}'.format(save_path))

log_fname = os.path.join(save_path, 'vis_log_{:s}.txt'.format(getDateTime()))
print('Saving log to: {}'.format(log_fname))

if selective_mode:
    label_diff = int(255.0 / n_classes)
else:
    label_diff = int(255.0 / (n_classes - 1))

print('label_diff: {}'.format(label_diff))

n_frames = end_id - start_id + 1

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
fw_sum = np.zeros((n_classes,))

print_diff = int(n_frames * 0.01)

avg_mean_acc_ice = avg_mean_acc_ice_1 = avg_mean_acc_ice_2 = 0
avg_mean_IU_ice = avg_mean_IU_ice_1 = avg_mean_IU_ice_2 = 0

skip_mean_acc_ice_1 = skip_mean_acc_ice_2 = 0
skip_mean_IU_ice_1 = skip_mean_IU_ice_2 = 0
_pause = 1
labels_img = None

for img_id in range(start_id, end_id + 1):

    # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
    img_fname = src_files[img_id]
    img_fname_no_ext = os.path.splitext(img_fname)[0]

    if stitch or show_img:
        src_img_fname = os.path.join(images_path, img_fname)
        src_img = imread(src_img_fname)
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

    if not labels_path:
        stitched = src_img
    else:
        labels_img_fname = os.path.join(labels_path, img_fname_no_ext + '.{}'.format(labels_ext))
        labels_img_orig = imread(labels_img_fname)
        if labels_img_orig is None:
            raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))
        _, src_width = labels_img_orig.shape[:2]

        if len(labels_img_orig.shape) == 3:
            labels_img_orig = np.squeeze(labels_img_orig[:, :, 0])

        if show_img:
            cv2.imshow('labels_img_orig', labels_img_orig)

        if normalize_labels:
            if selective_mode:
                selective_idx = (labels_img_orig == 255)
                print('labels_img_orig.shape: {}'.format(labels_img_orig.shape))
                print('selective_idx count: {}'.format(np.count_nonzero(selective_idx)))
                labels_img_orig[selective_idx] = n_classes
                if show_img:
                    cv2.imshow('labels_img_orig norm', labels_img_orig)
            labels_img = (labels_img_orig.astype(np.float64) * label_diff).astype(np.uint8)
        else:
            labels_img = np.copy(labels_img_orig)

        if len(labels_img.shape) != 3:
            labels_img = np.stack((labels_img, labels_img, labels_img), axis=2)

        if stitch:
            stitched = np.concatenate((src_img, labels_img), axis=1)

        if eval_mode:
            seg_img_fname = os.path.join(seg_path, img_fname_no_ext + '.{}'.format(seg_ext))
            seg_img = imread(seg_img_fname)
            if seg_img is None:
                raise SystemError('Segmentation image could not be read from: {}'.format(seg_img_fname))

            if len(seg_img.shape) == 3:
                seg_img = np.squeeze(seg_img[:, :, 0])

            eval_cl, _ = eval.extract_classes(seg_img)
            gt_cl, _ = eval.extract_classes(labels_img_orig)

            if eval_cl != gt_cl:
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

            _fw_total = np.sum(_fw)

            # print('_fw: {}'.format(_fw))
            # print('_fw_total: {}'.format(_fw_total))

            _fw_frac = np.array(_fw) / float(_fw_total)

            print('_fw_frac: {}'.format(_fw_frac))

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

            if stitch and stitch_seg:
                stitched = np.concatenate((stitched, seg_img), axis=1)
            if not stitch and show_img:
                cv2.imshow('seg_img', seg_img)

    if stitch:
        if save_stitched:
            seg_save_path = os.path.join(save_path, '{}.{}'.format(img_fname_no_ext, out_ext))
            imsave(seg_save_path, stitched)
        if show_img:
            cv2.imshow('stitched', stitched)
    else:
        if show_img:
            cv2.imshow('src_img', src_img)
            if labels_path:
                cv2.imshow('labels_img', labels_img)

    if show_img:
        k = cv2.waitKey(1 - _pause)
        if k == 27:
            sys.exit(0)
        elif k == 32:
            _pause = 1 - _pause
    img_done = img_id - start_id + 1
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

    fw_sum_total = np.sum(fw_sum)
    fw_sum_frac = fw_sum / float(fw_sum_total)

    print('fw_sum_total: {}'.format(fw_sum_total))
    print('fw_sum_frac: {}'.format(fw_sum_frac))
