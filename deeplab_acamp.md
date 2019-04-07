<!-- MarkdownTOC -->

- [bear_1_1_masks_448x448_ar_1p0](#bear11_masks_448x448_ar_1p0)
    - [build_data       @ bear_1_1_masks_448x448_ar_1p0](#build_data__bear11_masks_448x448_ar_1p0)
    - [train       @ bear_1_1_masks_448x448_ar_1p0](#train__bear11_masks_448x448_ar_1p0)
    - [vis       @ bear_1_1_masks_448x448_ar_1p0](#vis__bear11_masks_448x448_ar_1p0)
- [bear_1_1_500x500_10](#bear11_500x500_10)
    - [build_data       @ bear_1_1_500x500_10](#build_data__bear11_500x500_10)
    - [train       @ bear_1_1_500x500_10](#train__bear11_500x500_10)
- [visualize](#visualize)
    - [bear_1_1       @ visualize](#bear11__visualize)

<!-- /MarkdownTOC -->

<a id="bear11_masks_448x448_ar_1p0"></a>
# bear_1_1_masks_448x448_ar_1p0

<a id="build_data__bear11_masks_448x448_ar_1p0"></a>
## build_data       @ bear_1_1_masks_448x448_ar_1p0

python datasets/build_617_data.py --db_root_dir=/data/acamp/acamp20k/ --db_dir=bear_1_1_masks_448x448_ar_1p0 --image_format=jpg --label_format=png --output_dir=deeplab_acamp

<a id="train__bear11_masks_448x448_ar_1p0"></a>
## train       @ bear_1_1_masks_448x448_ar_1p0

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=448 --train_crop_size=448 --train_batch_size=2 --dataset=deeplab_acamp --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/deeplab_acamp/bear_1_1_masks_448x448_ar_1p0 --dataset_dir=/data/acamp/acamp20k/deeplab_acamp/tfrecord --train_split=bear_1_1_masks_448x448_ar_1p0 --num_clones=1

<a id="vis__bear11_masks_448x448_ar_1p0"></a>
## vis       @ bear_1_1_masks_448x448_ar_1p0

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=448 --vis_crop_size=448 --dataset="deeplab_acamp" --checkpoint_dir=log/deeplab_acamp/bear_1_1_masks_448x448_ar_1p0 --vis_logdir=log/deeplab_acamp/bear_1_1_masks_448x448_ar_1p0 --dataset_dir=/data/acamp/acamp20k/deeplab_acamp/tfrecord --vis_split=bear_1_1_masks_448x448_ar_1p0 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/acamp/acamp20k/bear_1_1_masks_448x448_ar_1p0/images --seg_path=log/deeplab_acamp/bear_1_1_masks_448x448_ar_1p0/raw --save_path=log/deeplab_acamp/bear_1_1_masks_448x448_ar_1p0/vis --n_classes=2 --start_id=0 --end_id=-1 --images_ext=jpg

python3 ../stitchSubPatchDataset.py src_path=/data/acamp/acamp20k/bear_1_1_masks_448x448_ar_1p0/images patch_seq_path=log/deeplab_acamp/bear_1_1_masks_448x448_ar_1p0/raw stitched_seq_path=log/deeplab_acamp/bear_1_1_masks_448x448_ar_1p0 patch_height=448 patch_width=448 start_id=0 end_id=-1  show_img=0 stacked=0 method=-1 normalize_patches=0 img_ext=jpg out_ext=png width=1920 height=1080 n_classes=2

<a id="bear11_500x500_10"></a>
# bear_1_1_500x500_10

<a id="build_data__bear11_500x500_10"></a>
## build_data       @ bear_1_1_500x500_10

python datasets/build_617_data.py --db_root_dir=/data/acamp/acamp20k/masks --db_dir=bear_1_1_500x500_10 --image_format=jpg --label_format=png --output_dir=deeplab_acamp

<a id="train__bear11_500x500_10"></a>
## train       @ bear_1_1_500x500_10

CUDA_VISIBLE_DEVICES=2 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=500 --train_crop_size=500 --train_batch_size=2 --dataset=deeplab_acamp --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/deeplab_acamp/bear_1_1_500x500_10 --dataset_dir=/data/acamp/acamp20k/deeplab_acamp/tfrecord --train_split=bear_1_1_500x500_10 --num_clones=1

<a id="visualize"></a>
# visualize

<a id="bear11__visualize"></a>
## bear_1_1       @ visualize

python3 visualize_masks.py img_paths=bear_1_1 img_root_dir=/data/acamp/acamp20k/bear mask_paths=/results/deeplab/acamp_bear_1_1_masks_448x448_ar_1p0_png_grs_190325_155703 map_to_bbox=1  fixed_ar=1 save_video=1
