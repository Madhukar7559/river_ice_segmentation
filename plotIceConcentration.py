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

parser.add_argument("--out_path", type=str, default='')

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

out_path = args.out_path


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

if not out_path:
    out_path = labels_path+'_conc'
    if not os.path.isdir(labels_path):
        os.makedirs(out_path)
    print('Writing concentration data to {}'.format(out_path))


if not save_path:
    save_path = os.path.join(os.path.dirname(images_path), 'ice_concentration')

if not os.path.isdir(save_path):
    os.makedirs(save_path)

if stitch and save_stitched:
    print('Saving ice_concentration plots to: {}'.format(save_path))

log_fname = os.path.join(save_path, 'vis_log_{:s}.txt'.format(getDateTime()))
print('Saving log to: {}'.format(log_fname))

if selective_mode:
    label_diff = int(255.0 / n_classes)
else:
    label_diff = int(255.0 / (n_classes - 1))

print('label_diff: {}'.format(label_diff))

n_frames = end_id - start_id + 1


print_diff = int(n_frames * 0.01)

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

        if len(labels_img.shape) == 3:
            labels_img = labels_img[:, :, 0].squeeze()
            # labels_img = np.stack((labels_img, labels_img, labels_img), axis=2)

        conc_data_x = np.asarray(range(src_width), dtype=np.float64)
        conc_data_y = np.zeros((src_width,), dtype=np.float64)
        for i in range(src_width):
            curr_pix = np.squeeze(labels_img[:, i])
            ice_pix = curr_pix[curr_pix != 0]

            conc_data_y[i] = len(ice_pix) / float(src_height)

        conc_data = np.concatenate([conc_data_x, conc_data_y], axis=1)

        conc_data_fname = os.path.join(out_path, img_fname_no_ext + '.txt')
        np.savetxt(conc_data_fname, conc_data, fmt='%.6f')


