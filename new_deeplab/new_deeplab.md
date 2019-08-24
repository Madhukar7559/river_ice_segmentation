
# build_data

## voc2012       @ build_data

CUDA_VISIBLE_DEVICES=2 python datasets/build_voc2012_data.py --image_folder=/data/voc2012/JPEGImages --semantic_segmentation_folder=/data/voc2012/SegmentationClass --list_folder=/data/voc2012/ImageSets/Segmentation --image_format="jpg" --output_dir=/data/voc2012/tfrecord

## ade20k       @ build_data

CUDA_VISIBLE_DEVICES=2 python datasets/build_ade20k_data.py --train_image_folder=/data/ade20k/images/training/ --train_image_label_folder=/data/ade20k/annotations/training/ --val_image_folder=/data/ade20k/images/validation/ --val_image_label_folder=/data/ade20k/annotations/validation/ --output_dir=/data/ade20k/tfrecord

# hnasnet

## voc2012       @ hnasnet

CUDA_VISIBLE_DEVICES=0 python3 new_deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="nas_hnasnet" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size="513,513" --train_batch_size=2 --dataset=pascal_voc_seg --train_logdir=log/pascal_voc_seg/nas_hnasnet --dataset_dir=/data/voc2012/tfrecord --train_split=train --num_clones=1 --add_image_level_feature=0

## ade20k       @ hnasnet

CUDA_VISIBLE_DEVICES=2 python3 new_deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="nas_hnasnet" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size="513,513" --train_batch_size=4 --dataset=ade20k --train_logdir=log/ade20k/nas_hnasnet --dataset_dir=/data/ade20k/tfrecord --train_split=train --num_clones=1 --add_image_level_feature=0 --min_resize_value=513 --max_resize_value=513 --resize_factor=16 

# resnet_v1_101_beta

<a id="32___64_0_"></a>
## voc2012       @ resnet_v1_101_beta

CUDA_VISIBLE_DEVICES=1 python3 new_deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant=resnet_v1_101_beta --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size="513,513" --train_batch_size=2 --dataset=pascal_voc_seg --train_logdir=log/pascal_voc_seg/resnet_v1_101_beta --dataset_dir=/data/voc2012/tfrecord --train_split=train --num_clones=1

<a id="32___64_0_"></a>
## ade20k       @ resnet_v1_101_beta

CUDA_VISIBLE_DEVICES=1 python3 new_deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant=resnet_v1_101_beta --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size="513,513" --train_batch_size=4 --dataset=ade20k --train_logdir=log/ade20k/resnet_v1_101_beta --dataset_dir=/data/ade20k/tfrecord --train_split=train --num_clones=1 --min_resize_value=513 --max_resize_value=513 --resize_factor=16

# 640_hnasnet

<a id="32___64_0_"></a>
## 32       @ 640_hnasnet

CUDA_VISIBLE_DEVICES=0 python3 new_deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="nas_hnasnet" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size="640,640" --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_31_640_640_64_256_rot_15_345_4_flip --num_clones=1 --add_image_level_feature=0

<a id="vis___32_640_"></a>
### vis       @ 32/640_hnasnet

CUDA_VISIBLE_DEVICES=0 python3 new_deeplab_vis.py --logtostderr --model_variant="resnet_v1_101_beta" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="stitched___no_aug_32_64_0_"></a>
#### stitched       @ vis/32/640_hnasnet

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1 --normalize_labels=1

# 640_resnet_v1_101_beta

<a id="32___64_0_"></a>
## 32       @ 640_resnet_v1_101_beta

CUDA_VISIBLE_DEVICES=2 python3 new_deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="resnet_v1_101_beta" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size="640,640" --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_31_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis___32_640_"></a>
### vis       @ 32/640_resnet_v1_101_beta

CUDA_VISIBLE_DEVICES=0 python3 new_deeplab_vis.py --logtostderr --model_variant="resnet_v1_101_beta" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug___32_640_"></a>
### no_aug       @ 32/640_resnet_v1_101_beta

CUDA_VISIBLE_DEVICES=0 python3 new_deeplab_vis.py --logtostderr --model_variant="resnet_v1_101_beta" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_32_64_0_"></a>
#### stitched       @ no_aug/32/640_resnet_v1_101_beta

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1 --normalize_labels=1

<a id="yun00001___32_640_"></a>
### YUN00001       @ 32/640_resnet_v1_101_beta

CUDA_VISIBLE_DEVICES=0 python3 new_deeplab_vis.py --logtostderr --model_variant="resnet_v1_101_beta" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/YUN00001_0_8999_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=YUN00001_0_8999_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/YUN00001_0_8999_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/YUN00001 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png out_ext=mkv width=1920 height=1080

<a id="yun00001_3600___32_640_"></a>
### YUN00001_3600       @ 32/640_resnet_v1_101_beta

CUDA_VISIBLE_DEVICES=0 python3 new_deeplab_vis.py --logtostderr --model_variant="resnet_v1_101_beta" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/YUN00001_3600_0_3599_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=YUN00001_3600_0_3599_640_640_640_640 --vis_batch_size=25 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/YUN00001_3600_0_3599_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/YUN00001_3600 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080