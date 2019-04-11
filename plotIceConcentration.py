import argparse, os, sys
import numpy as np
from scipy.misc.pilutil import imread, imsave
from matplotlib import pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pprint import pprint

import densenet.evaluation.eval_segm as eval
from densenet.utils import readData, getDateTime, print_and_write, resizeAR
from dictances import bhattacharyya, euclidean, mae, mse


def getPlotImage(data, cols, title, line_labels, x_label, y_label):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    n_data = len(data)
    for i in range(n_data):
        ax.plot(data[i], color=cols[i], label=line_labels[i])
    ax.set_title(title)
    ax.legend()
    ax.grid(1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_ylim(0, 100)

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plot_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
        int(height), int(width), 3)

    return plot_img


def str_to_list(_str, _type=str, _sep=','):
    if _sep not in _str:
        _str += _sep
    return [k for k in list(map(_type, _str.split(_sep))) if k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)

    parser.add_argument("--images_path", type=str)
    parser.add_argument("--images_ext", type=str, default='png')
    parser.add_argument("--labels_path", type=str, default='')
    parser.add_argument("--labels_ext", type=str, default='png')
    parser.add_argument("--seg_paths", type=str_to_list, default=[])
    parser.add_argument("--seg_ext", type=str, default='png')
    parser.add_argument("--seg_root", type=str, default='')

    parser.add_argument("--seg_labels", type=str_to_list, default=[])

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
    parser.add_argument("--ice_type", type=int, default=0, help='0: combined, 1: anchor, 2: frazil')

    args = parser.parse_args()

    images_path = args.images_path
    images_ext = args.images_ext
    labels_path = args.labels_path
    labels_ext = args.labels_ext

    out_path = args.out_path

    seg_paths = args.seg_paths
    seg_ext = args.seg_ext
    seg_root = args.seg_root

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

    seg_labels = args.seg_labels

    ice_type = args.ice_type

    ice_types = {
        0: 'combined ice',
        1: 'anchor ice',
        2: 'frazil ice',
    }

    ice_type_str = ice_types[ice_type]

    src_files, src_labels_list, total_frames = readData(images_path, images_ext, labels_path,
                                                        labels_ext)
    if end_id < start_id:
        end_id = total_frames - 1

    cols = [(1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    if not out_path:
        out_path = labels_path + '_conc'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        print('Writing concentration data to {}'.format(out_path))

    if seg_paths:
        n_seg_paths = len(seg_paths)
        n_seg_labels = len(seg_labels)

        if n_seg_paths != n_seg_labels:
            raise IOError('Mismatch between n_seg_labels: {} and n_seg_paths: {}'.format(
                n_seg_labels, n_seg_paths
            ))

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

    n_cols = len(cols)

    plot_y_label = '{} concentration (%)'.format(ice_type_str)
    plot_x_label = 'distance in pixels from left edge'

    dists = {}

    for _label in seg_labels:
        dists[_label] = {
            # 'bhattacharyya': [],
            'euclidean': [],
            'mae': [],
            'mse': [],
            # 'frobenius': [],
        }

    for img_id in range(start_id, end_id + 1):

        # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
        img_fname = src_files[img_id]
        img_fname_no_ext = os.path.splitext(img_fname)[0]

        src_img_fname = os.path.join(images_path, img_fname)
        src_img = imread(src_img_fname)
        if src_img is None:
            raise SystemError('Source image could not be read from: {}'.format(src_img_fname))

        try:
            src_height, src_width = src_img.shape[:2]
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
                labels_img = (labels_img_orig.astype(np.float64) * label_diff).astype(np.uint8)
            else:
                labels_img = np.copy(labels_img_orig)

            if len(labels_img.shape) == 3:
                labels_img = labels_img[:, :, 0].squeeze()

            conc_data_x = np.asarray(range(src_width), dtype=np.float64)
            conc_data_y = np.zeros((src_width,), dtype=np.float64)

            for i in range(src_width):
                curr_pix = np.squeeze(labels_img_orig[:, i])
                if ice_type == 0:
                    ice_pix = curr_pix[curr_pix != 0]
                else:
                    ice_pix = curr_pix[curr_pix == ice_type]

                conc_data_y[i] = (len(ice_pix) / float(src_height))*100.0

            conc_data = np.zeros((src_width, 2), dtype=np.float64)
            conc_data[:, 0] = conc_data_x
            conc_data[:, 1] = conc_data_y

            plot_data_y = [conc_data_y, ]
            plot_cols_y = [(0, 1, 0), ]

            gt_dict = {conc_data_x[i]: conc_data_y[i] for i in range(src_width)}

            # gt_cl, _ = eval.extract_classes(labels_img_orig)
            # print('gt_cl: {}'.format(gt_cl))

            for seg_id, seg_path in enumerate(seg_paths):
                seg_img_fname = os.path.join(seg_path, img_fname_no_ext + '.{}'.format(seg_ext))
                seg_img_orig = imread(seg_img_fname)

                _label = seg_labels[seg_id]

                if seg_img_orig is None:
                    raise SystemError('Seg image could not be read from: {}'.format(seg_img_fname))
                _, src_width = seg_img_orig.shape[:2]

                if len(seg_img_orig.shape) == 3:
                    seg_img_orig = np.squeeze(seg_img_orig[:, :, 0])

                if seg_img_orig.max() > n_classes - 1:
                    seg_img = (seg_img_orig.astype(np.float64) / label_diff).astype(np.uint8)
                else:
                    seg_img = seg_img_orig

                # eval_cl, _ = eval.extract_classes(seg_img)
                # print('eval_cl: {}'.format(eval_cl))

                if show_img:
                    cv2.imshow('seg_img_orig', seg_img_orig)

                if len(seg_img.shape) == 3:
                    seg_img = seg_img[:, :, 0].squeeze()

                conc_data_x = np.asarray(range(src_width), dtype=np.float64)
                conc_data_y = np.zeros((src_width,), dtype=np.float64)
                for i in range(src_width):
                    curr_pix = np.squeeze(seg_img[:, i])
                    if ice_type == 0:
                        ice_pix = curr_pix[curr_pix != 0]
                    else:
                        ice_pix = curr_pix[curr_pix == ice_type]
                    conc_data_y[i] = (len(ice_pix) / float(src_height)) * 100.0

                plot_cols_y.append(cols[seg_id % n_cols])
                plot_data_y.append(conc_data_y)

                seg_dict = {conc_data_x[i]: conc_data_y[i] for i in range(src_width)}

                # dists['bhattacharyya'].append(bhattacharyya(gt_dict, seg_dict))
                dists[_label]['euclidean'].append(euclidean(gt_dict, seg_dict))
                dists[_label]['mse'].append(mse(gt_dict, seg_dict))
                dists[_label]['mae'].append(mae(gt_dict, seg_dict))
                # dists['frobenius'].append(np.linalg.norm(conc_data_y - plot_data_y[0]))

            # conc_data = np.concatenate([conc_data_x, conc_data_y], axis=1)

            plot_title = 'ice concentration'
            plot_labels = ['GT', ] + seg_labels
            plot_img = getPlotImage(plot_data_y, plot_cols_y, plot_title, plot_labels,
                                    plot_x_label, plot_y_label)
            plot_img = resizeAR(plot_img, src_width, src_height, bkg_col=255)

            # plt.plot(conc_data_x, conc_data_y)
            # plt.show()

            # conc_data_fname = os.path.join(out_path, img_fname_no_ext + '.txt')
            # np.savetxt(conc_data_fname, conc_data, fmt='%.6f')

            stitched = np.concatenate((src_img, plot_img), axis=1)

            stitched = resizeAR(stitched, width=1280)

            # print('dists: {}'.format(dists))

            cv2.imshow('stitched', stitched)
            k = cv2.waitKey(1 - _pause)
            if k == 27:
                break
            elif k == 32:
                _pause = 1 - _pause

    mean_dists = {}
    mae_data = []
    for _label in dists:
        _dists = dists[_label]
        mae_data.append(_dists['mae'])
        mean_dists[_label]= {k:np.mean(_dists[k]) for k in _dists}

    print('mean_dists: {}'.format(mean_dists))
    pprint(mean_dists)


    plot_img = getPlotImage(plot_data_y, plot_cols_y, plot_title, plot_labels,
                            plot_x_label, plot_y_label)
    plt.show()

if __name__ == '__main__':
    main()
