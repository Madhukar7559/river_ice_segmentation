<!-- MarkdownTOC -->

- [setup](#setup)
- [256](#256)
   - [build_data       @ 256](#build_data__256)
   - [50       @ 256](#50__256)
      - [output_stride_16       @ 50/256](#output_stride_16__50256)
         - [eval       @ output_stride_16/50/256](#eval__output_stride_1650256)
         - [vis       @ output_stride_16/50/256](#vis__output_stride_1650256)
         - [validation       @ output_stride_16/50/256](#validation__output_stride_1650256)
         - [stitching       @ output_stride_16/50/256](#stitching__output_stride_1650256)
      - [output_stride_8       @ 50/256](#output_stride_8__50256)
         - [eval       @ output_stride_8/50/256](#eval__output_stride_850256)
         - [vis       @ output_stride_8/50/256](#vis__output_stride_850256)
         - [validation       @ output_stride_8/50/256](#validation__output_stride_850256)
   - [32       @ 256](#32__256)
      - [eval       @ 32/256](#eval__32256)
      - [vis       @ 32/256](#vis__32256)
      - [validation       @ 32/256](#validation__32256)
- [384](#384)
   - [build_data       @ 384](#build_data__384)
   - [50       @ 384](#50__384)
      - [batch_size_6       @ 50/384](#batch_size_6__50384)
         - [eval       @ batch_size_6/50/384](#eval__batch_size_650384)
         - [vis       @ batch_size_6/50/384](#vis__batch_size_650384)
         - [validation       @ batch_size_6/50/384](#validation__batch_size_650384)
      - [batch_size_8       @ 50/384](#batch_size_8__50384)
         - [eval       @ batch_size_8/50/384](#eval__batch_size_850384)
         - [vis       @ batch_size_8/50/384](#vis__batch_size_850384)
   - [stride_8       @ 384](#stride_8__384)
   - [32       @ 384](#32__384)
      - [eval       @ 32/384](#eval__32384)
      - [vis       @ 32/384](#vis__32384)
      - [validation       @ 32/384](#validation__32384)
- [512](#512)
   - [build_data       @ 512](#build_data__512)
   - [50       @ 512](#50__512)
      - [batch_size_6       @ 50/512](#batch_size_6__50512)
         - [eval       @ batch_size_6/50/512](#eval__batch_size_650512)
         - [vis       @ batch_size_6/50/512](#vis__batch_size_650512)
         - [validation       @ batch_size_6/50/512](#validation__batch_size_650512)
      - [batch_size_2       @ 50/512](#batch_size_2__50512)
         - [vis       @ batch_size_2/50/512](#vis__batch_size_250512)
         - [validation       @ batch_size_2/50/512](#validation__batch_size_250512)
   - [32       @ 512](#32__512)
      - [eval       @ 32/512](#eval__32512)
      - [vis       @ 32/512](#vis__32512)
      - [validation       @ 32/512](#validation__32512)
- [640](#640)
   - [build_data       @ 640](#build_data__640)
      - [0_3       @ build_data/640](#0_3__build_data640)
      - [0_3_non_aug       @ build_data/640](#03non_aug__build_data640)
         - [sel_2       @ 0_3_non_aug/build_data/640](#sel_2__03non_augbuild_data640)
         - [sel_10       @ 0_3_non_aug/build_data/640](#sel_10__03non_augbuild_data640)
         - [sel_100       @ 0_3_non_aug/build_data/640](#sel_100__03non_augbuild_data640)
         - [sel_1000       @ 0_3_non_aug/build_data/640](#sel_1000__03non_augbuild_data640)
      - [0_7       @ build_data/640](#0_7__build_data640)
      - [0_15       @ build_data/640](#0_15__build_data640)
      - [0_23       @ build_data/640](#0_23__build_data640)
      - [0_31       @ build_data/640](#0_31__build_data640)
      - [0_49       @ build_data/640](#0_49__build_data640)
      - [32_49       @ build_data/640](#32_49__build_data640)
      - [4_49_no_aug       @ build_data/640](#4_49_no_aug__build_data640)
      - [32_49_no_aug       @ build_data/640](#32_49_no_aug__build_data640)
      - [validation_0_20       @ build_data/640](#validation020__build_data640)
      - [YUN00001_0_239       @ build_data/640](#yun000010239__build_data640)
      - [YUN00001_3600       @ build_data/640](#yun00001_3600__build_data640)
      - [YUN00001_0_8999       @ build_data/640](#yun0000108999__build_data640)
      - [20160122_YUN00002_700_2500       @ build_data/640](#20160122_yun00002_700_2500__build_data640)
      - [20160122_YUN00020_2000_3800       @ build_data/640](#20160122_yun00020_2000_3800__build_data640)
      - [20161203_Deployment_1_YUN00001_900_2700       @ build_data/640](#20161203_deployment1yun00001_900_2700__build_data640)
      - [20161203_Deployment_1_YUN00002_1800       @ build_data/640](#20161203_deployment1yun00002_1800__build_data640)
   - [50       @ 640](#50__640)
      - [eval       @ 50/640](#eval__50640)
      - [vis       @ 50/640](#vis__50640)
         - [stitching       @ vis/50/640](#stitching__vis50640)
      - [validation       @ 50/640](#validation__50640)
         - [stitching       @ validation/50/640](#stitching__validation50640)
         - [vis       @ validation/50/640](#vis__validation50640)
         - [zip       @ validation/50/640](#zip__validation50640)
      - [video       @ 50/640](#video__50640)
         - [stitching       @ video/50/640](#stitching__video50640)
   - [32       @ 640](#32__640)
      - [vis_png       @ 32/640](#vis_png__32640)
         - [20160122_YUN00002_700_2500       @ vis_png/32/640](#20160122_yun00002_700_2500__vis_png32640)
         - [20160122_YUN00020_2000_3800       @ vis_png/32/640](#20160122_yun00020_2000_3800__vis_png32640)
         - [20161203_Deployment_1_YUN00001_900_2700       @ vis_png/32/640](#20161203_deployment1yun00001_900_2700__vis_png32640)
         - [20161203_Deployment_1_YUN00002_1800       @ vis_png/32/640](#20161203_deployment1yun00002_1800__vis_png32640)
         - [YUN00001_3600       @ vis_png/32/640](#yun00001_3600__vis_png32640)
   - [4       @ 640](#4__640)
      - [continue_40787       @ 4/640](#continue_40787__4640)
      - [vis       @ 4/640](#vis__4640)
      - [no_aug       @ 4/640](#no_aug__4640)
         - [stitched       @ no_aug/4/640](#stitched__no_aug4640)
      - [no_aug_4_49       @ 4/640](#no_aug449__4640)
         - [stitched       @ no_aug_4_49/4/640](#stitched__no_aug4494640)
   - [8       @ 640](#8__640)
      - [vis       @ 8/640](#vis__8640)
      - [no_aug       @ 8/640](#no_aug__8640)
         - [stitched       @ no_aug/8/640](#stitched__no_aug8640)
   - [16       @ 640](#16__640)
      - [vis       @ 16/640](#vis__16640)
      - [no_aug       @ 16/640](#no_aug__16640)
         - [stitched       @ no_aug/16/640](#stitched__no_aug16640)
   - [16_rt       @ 640](#16_rt__640)
      - [no_aug       @ 16_rt/640](#no_aug__16_rt640)
         - [stitched       @ no_aug/16_rt/640](#stitched__no_aug16_rt640)
   - [16_rt_3       @ 640](#16_rt_3__640)
      - [no_aug       @ 16_rt_3/640](#no_aug__16_rt_3640)
         - [stitched       @ no_aug/16_rt_3/640](#stitched__no_aug16_rt_3640)
   - [24       @ 640](#24__640)
      - [vis       @ 24/640](#vis__24640)
      - [no_aug       @ 24/640](#no_aug__24640)
         - [stitched       @ no_aug/24/640](#stitched__no_aug24640)
   - [32       @ 640](#32__640-1)
      - [vis       @ 32/640](#vis__32640)
      - [no_aug       @ 32/640](#no_aug__32640)
         - [stitched       @ no_aug/32/640](#stitched__no_aug32640)
      - [YUN00001       @ 32/640](#yun00001__32640)
      - [YUN00001_3600       @ 32/640](#yun00001_3600__32640)
   - [4__non_aug       @ 640](#4__non_aug__640)
      - [sel_2       @ 4__non_aug/640](#sel_2__4__non_aug640)
      - [sel_10       @ 4__non_aug/640](#sel_10__4__non_aug640)
      - [sel_100       @ 4__non_aug/640](#sel_100__4__non_aug640)
      - [sel_1000       @ 4__non_aug/640](#sel_1000__4__non_aug640)
         - [rt       @ sel_1000/4__non_aug/640](#rt__sel_10004__non_aug640)
         - [rt2       @ sel_1000/4__non_aug/640](#rt2__sel_10004__non_aug640)
- [800](#800)
   - [build_data       @ 800](#build_data__800)
      - [50       @ build_data/800](#50__build_data800)
      - [32       @ build_data/800](#32__build_data800)
      - [18_-_test       @ build_data/800](#18-test__build_data800)
      - [validation_0_20_800_800_800_800       @ build_data/800](#validation020_800_800_800_800__build_data800)
      - [YUN00001_0_239_800_800_800_800       @ build_data/800](#yun000010239_800_800_800_800__build_data800)
      - [4       @ build_data/800](#4__build_data800)
   - [50       @ 800](#50__800)
      - [eval       @ 50/800](#eval__50800)
      - [vis       @ 50/800](#vis__50800)
         - [stitching       @ vis/50/800](#stitching__vis50800)
      - [validation       @ 50/800](#validation__50800)
         - [stitching       @ validation/50/800](#stitching__validation50800)
         - [vis       @ validation/50/800](#vis__validation50800)
   - [4       @ 800](#4__800)

<!-- /MarkdownTOC -->

<a id="setup"></a>
# setup 

pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp35-cp35m-linux_x86_64.whl
pip2 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp27-none-linux_x86_64.whl
python remove_gt_colormap.py --original_gt_folder="/home/abhineet/N/Datasets/617/images/training_256_256_25_100_rot_15_90_flip/labels" --output_dir="/home/abhineet/N/Datasets/617/images/training_256_256_25_100_rot_15_90_flip/labels_raw"

lftp -e "cls -1 > _list; exit" "http://download.tensorflow.org/models"

<a id="256"></a>
# 256

<a id="build_data__256"></a>
## build_data       @ 256

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_256_256_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_256_256_25_100_rot_15_125_235_345_flip --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_256_256_25_100_rot_15_125_235_345_flip --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_256_256_256_256 --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_256_256_256_256 --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip --create_dummy_labels=1


<a id="50__256"></a>
## 50       @ 256

<a id="output_stride_16__50256"></a>
### output_stride_16       @ 50/256

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=256 --train_crop_size=256 --train_batch_size=10 --dataset=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --train_split=training_0_49_256_256_25_100_rot_15_345_4_flip --num_clones=2


<a id="eval__output_stride_1650256"></a>
#### eval       @ output_stride_16/50/256

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=256 --eval_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --eval_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --eval_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip

miou_1.0[0.802874804]


<a id="vis__output_stride_1650256"></a>
#### vis       @ output_stride_16/50/256

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip/raw --save_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__output_stride_1650256"></a>
#### validation       @ output_stride_16/50/256


CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=256 --vis_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/vis --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=validation_0_20_256_256_256_256 --vis_batch_size=10 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="stitching__output_stride_1650256"></a>
#### stitching       @ output_stride_16/50/256

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_20_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=10 patch_seq_type=labels_deeplab_xception show_img=0 stacked=1 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_20_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=10 patch_seq_type=images show_img=0 stacked=1 method=1


<a id="output_stride_8__50256"></a>
### output_stride_8       @ 50/256

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --train_crop_size=256 --train_crop_size=256 --train_batch_size=10 --dataset=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8 --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --train_split=training_0_49_256_256_25_100_rot_15_345_4_flip --num_clones=2

<a id="eval__output_stride_850256"></a>
#### eval       @ output_stride_8/50/256

CUDA_VISIBLE_DEVICES=0 python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --eval_crop_size=256 --eval_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8 --eval_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/eval --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --eval_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip

miou_1.0[0.799446404]
miou_1.0[0.799446404]

<a id="vis__output_stride_850256"></a>
#### vis       @ output_stride_8/50/256

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

zr vis_xception_0_49_stride8_training_32_49_256_256_25_100_rot_15_125_235_345_flip log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip/raw --save_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis_xception_0_49_stride8 --n_classes=3 --start_id=0 --end_id=-1


<a id="validation__output_stride_850256"></a>
#### validation       @ output_stride_8/50/256

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --vis_crop_size=256 --vis_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/vis --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=validation_0_20_256_256_256_256 --vis_batch_size=10 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="32__256"></a>
## 32       @ 256

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=256 --train_crop_size=256 --train_batch_size=10 --dataset=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip --tf_initial_checkpoint=pre_trained/xception_0_31/model.ckpt --train_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31 --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --train_split=training_0_31_256_256_25_100_rot_15_125_235_345_flip


<a id="eval__32256"></a>
### eval       @ 32/256

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=256 --eval_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31 --eval_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31/eval --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --eval_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip


<a id="vis__32256"></a>
### vis       @ 32/256

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip/raw --save_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis_xception_0_31 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__32256"></a>
### validation       @ 32/256

python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=256 --vis_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31/vis --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=validation_0_20_256_256_256_256 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

miou_1.0[0.735878468]

<a id="384"></a>
# 384

<a id="build_data__384"></a>
## build_data       @ 384

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_384_384_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_384_384_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_384_384_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_384_384_384_384 --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_384_384_384_384 --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip --create_dummy_labels=1


<a id="50__384"></a>
## 50       @ 384

<a id="batch_size_6__50384"></a>
### batch_size_6       @ 50/384

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_384_384_25_100_rot_15_345_4_flip --num_clones=2

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_attempt_2 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_384_384_25_100_rot_15_345_4_flip --num_clones=2


<a id="eval__batch_size_650384"></a>
#### eval       @ batch_size_6/50/384

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=384 --eval_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --eval_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_384_384_25_100_rot_15_345_4_flip --eval_batch_size=5


<a id="vis__batch_size_650384"></a>
#### vis       @ batch_size_6/50/384

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_384_384_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_384_384_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_384_384_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__batch_size_650384"></a>
#### validation       @ batch_size_6/50/384

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=384 --vis_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/vis --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_384_384_384_384 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="batch_size_8__50384"></a>
### batch_size_8       @ 50/384

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=8 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_384_384_25_100_rot_15_345_4_flip --num_clones=2

<a id="eval__batch_size_850384"></a>
#### eval       @ batch_size_8/50/384

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=384 --eval_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8 --eval_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8/eval --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_384_384_25_100_rot_15_345_4_flip --eval_batch_size=5


<a id="vis__batch_size_850384"></a>
#### vis       @ batch_size_8/50/384

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8/vis/training_32_49_384_384_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_384_384_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

zrq xception_0_49_batch_8_training_32_49_384_384_25_100_rot_15_345_4_flip log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8/vis/training_32_49_384_384_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/vis_xception_0_49_batch_8 --n_classes=3 --start_id=0 --end_id=-1

<a id="stride_8__384"></a>
## stride_8       @ 384

CUDA_VISIBLE_DEVICES=1 python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_stride_8 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_384_384_25_100_rot_15_345_4_flip --num_clones=2

<a id="32__384"></a>
## 32       @ 384

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_31_384_384_25_100_rot_15_345_4_flip --num_clones=2

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31_attempt2 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_31_384_384_25_100_rot_15_345_4_flip --num_clones=2

<a id="eval__32384"></a>
### eval       @ 32/384

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=384 --eval_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31 --eval_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31/eval --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_384_384_25_100_rot_15_345_4_flip --eval_batch_size=5


<a id="vis__32384"></a>
### vis       @ 32/384

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31/vis/training_32_49_384_384_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_384_384_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31/vis/training_32_49_384_384_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/vis_xception_0_31 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__32384"></a>
### validation       @ 32/384

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=384 --vis_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31/vis --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_384_384_384_384 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="512"></a>
# 512

<a id="build_data__512"></a>
## build_data       @ 512

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_512_512_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_512_512_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_512_512_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_512_512_512_512 --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_512_512_512_512 --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip --create_dummy_labels=1


<a id="50__512"></a>
## 50       @ 512

<a id="batch_size_6__50512"></a>
### batch_size_6       @ 50/512

CUDA_VISIBLE_DEVICES=1 python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=512 --train_crop_size=512 --train_batch_size=6 --dataset=training_0_31_49_512_512_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_512_512_25_100_rot_15_345_4_flip --num_clones=1


<a id="eval__batch_size_650512"></a>
#### eval       @ batch_size_6/50/512

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=512 --eval_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --eval_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_512_512_25_100_rot_15_345_4_flip --eval_batch_size=5

miou_1.0[0.723538] 


<a id="vis__batch_size_650512"></a>
#### vis       @ batch_size_6/50/512

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_512_512_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_512_512_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_512_512_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__batch_size_650512"></a>
#### validation       @ batch_size_6/50/512

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=512 --vis_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/vis --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_512_512_512_512 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="batch_size_2__50512"></a>
### batch_size_2       @ 50/512

CUDA_VISIBLE_DEVICES=1 python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=512 --train_crop_size=512 --train_batch_size=2 --dataset=training_0_31_49_512_512_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2 --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_512_512_25_100_rot_15_345_4_flip --num_clones=1

<a id="vis__batch_size_250512"></a>
#### vis       @ batch_size_2/50/512

CUDA_VISIBLE_DEVICES=1 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2/vis/training_32_49_512_512_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_512_512_25_100_rot_15_345_4_flip --vis_batch_size=10 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2/vis/training_32_49_512_512_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/vis_xception_0_49_batch_2 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__batch_size_250512"></a>
#### validation       @ batch_size_2/50/512

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=512 --vis_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2/vis --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_512_512_512_512 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0


<a id="32__512"></a>
## 32       @ 512

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=512 --train_crop_size=512 --train_batch_size=6 --dataset=training_0_31_49_512_512_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31 --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_31_512_512_25_100_rot_15_345_4_flip --num_clones=2

<a id="eval__32512"></a>
### eval       @ 32/512

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=512 --eval_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31 --eval_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31/eval --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_512_512_25_100_rot_15_345_4_flip --eval_batch_size=5

<a id="vis__32512"></a>
### vis       @ 32/512

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31/vis/training_32_49_512_512_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_512_512_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31/vis/training_32_49_512_512_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/vis_xception_0_31 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__32512"></a>
### validation       @ 32/512

python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=512 --vis_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31/vis --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_512_512_512_512 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="640"></a>
# 640

<a id="build_data__640"></a>
## build_data       @ 640

<a id="0_3__build_data640"></a>
### 0_3       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="03non_aug__build_data640"></a>
### 0_3_non_aug       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="sel_2__03non_augbuild_data640"></a>
#### sel_2       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640_sel_2 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="sel_10__03non_augbuild_data640"></a>
#### sel_10       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640_sel_10 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="sel_100__03non_augbuild_data640"></a>
#### sel_100       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640_sel_100 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="sel_1000__03non_augbuild_data640"></a>
#### sel_1000       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640_sel_1000 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_7__build_data640"></a>
### 0_7       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_7_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_15__build_data640"></a>
### 0_15       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_15_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_23__build_data640"></a>
### 0_23       @ build_data/640

CUDA_VISIBLE_DEVICES=2 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_23_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_31__build_data640"></a>
### 0_31       @ build_data/640

CUDA_VISIBLE_DEVICES=1 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_49__build_data640"></a>
### 0_49       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="32_49__build_data640"></a>
### 32_49       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="4_49_no_aug__build_data640"></a>
### 4_49_no_aug       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_4_49_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="32_49_no_aug__build_data640"></a>
### 32_49_no_aug       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="validation020__build_data640"></a>
### validation_0_20       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="yun000010239__build_data640"></a>
### YUN00001_0_239       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=1

<a id="yun00001_3600__build_data640"></a>
### YUN00001_3600       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_3600_0_3599_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="yun0000108999__build_data640"></a>
### YUN00001_0_8999       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_8999_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="20160122_yun00002_700_2500__build_data640"></a>
### 20160122_YUN00002_700_2500       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=20160122_YUN00002_700_2500_0_1799_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="20160122_yun00020_2000_3800__build_data640"></a>
### 20160122_YUN00020_2000_3800       @ build_data/640

CUDA_VISIBLE_DEVICES= python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=20160122_YUN00020_2000_3800_0_1799_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="20161203_deployment1yun00001_900_2700__build_data640"></a>
### 20161203_Deployment_1_YUN00001_900_2700       @ build_data/640

CUDA_VISIBLE_DEVICES= python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=20161203_Deployment_1_YUN00001_900_2700_0_1799_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="20161203_deployment1yun00002_1800__build_data640"></a>
### 20161203_Deployment_1_YUN00002_1800       @ build_data/640

CUDA_VISIBLE_DEVICES= python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=20161203_Deployment_1_YUN00002_1800_0_1799_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0



<a id="50__640"></a>
## 50       @ 640

CUDA_VISIBLE_DEVICES=1 python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_49_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="eval__50640"></a>
### eval       @ 50/640

CUDA_VISIBLE_DEVICES=1 python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=640 --eval_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --eval_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_640_640_64_256_rot_15_345_4_flip --eval_batch_size=5

miou_1.0[0.739234447]

<a id="vis__50640"></a>
### vis       @ 50/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="stitching__vis50640"></a>
#### stitching       @ vis/50/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/training_32_49_640_640_64_256_rot_15_345_4_flip patch_height=640 patch_width=640 start_id=32 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=1

<a id="validation__50640"></a>
### validation       @ 50/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="stitching__validation50640"></a>
#### stitching       @ validation/50/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640/raw patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

<a id="vis__validation50640"></a>
#### vis       @ validation/50/640

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="zip__validation50640"></a>
#### zip       @ validation/50/640

zrb training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_20_640_640_640_640_1_25 log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640/segmentation_results/img_XXX_* 1 25

training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_20_640_640_640_640_1_25_grs_201804221134.zip

zrb training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_20_640_640_640_640_raw_1_25 log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640/raw/img_XXX_* 1 25

training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_20_640_640_640_640_raw_1_25_grs_201804221136.zip 

zrb nazio deeplab/results/log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640/segmentation_results/img_XXX_* 1 10 -j

<a id="video__50640"></a>
### video       @ 50/640

CUDA_VISIBLE_DEVICES=1 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/YUN00001_0_239_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=YUN00001_0_239_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="stitching__video50640"></a>
#### stitching       @ video/50/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/YUN00001_0_239_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/YUN00001_0_239_640_640_640_640/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=1


python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1


<a id="32__640"></a>
## 32       @ 640

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=6 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_31_640_640_64_256_rot_15_345_4_flip --num_clones=2

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=640 --eval_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --eval_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/eval --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_640_640_64_256_rot_15_345_4_flip --eval_batch_size=5

python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/vis --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="vis_png__32640"></a>
### vis_png       @ 32/640

<a id="20160122_yun00002_700_2500__vis_png32640"></a>
#### 20160122_YUN00002_700_2500       @ vis_png/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00002_700_2500/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20160122_YUN00002_700_2500_0_1799_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20160122_YUN00002_700_2500 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="20160122_yun00020_2000_3800__vis_png32640"></a>
#### 20160122_YUN00020_2000_3800       @ vis_png/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20160122_YUN00020_2000_3800_0_1799_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20160122_YUN00020_2000_3800 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="20161203_deployment1yun00001_900_2700__vis_png32640"></a>
#### 20161203_Deployment_1_YUN00001_900_2700       @ vis_png/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20161203_Deployment_1_YUN00001_900_2700_0_1799_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20161203_Deployment_1_YUN00001_900_2700 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="20161203_deployment1yun00002_1800__vis_png32640"></a>
#### 20161203_Deployment_1_YUN00002_1800       @ vis_png/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20161203_Deployment_1_YUN00002_1800_0_1799_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20161203_Deployment_1_YUN00002_1800 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png

<a id="yun00001_3600__vis_png32640"></a>
#### YUN00001_3600       @ vis_png/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600_0_3599_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="4__640"></a>
## 4       @ 640

CUDA_VISIBLE_DEVICES=2 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="continue_40787__4640"></a>
### continue_40787       @ 4/640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/model.ckpt-40787 --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis__4640"></a>
### vis       @ 4/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__4640"></a>
### no_aug       @ 4/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug4640"></a>
#### stitched       @ no_aug/4/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug449__4640"></a>
### no_aug_4_49       @ 4/640

CUDA_VISIBLE_DEVICES=1 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_4_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_4_49_640_640_640_640/images --labels_path=/data/617/images/training_4_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug4494640"></a>
#### stitched       @ no_aug_4_49/4/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_4_49/images --labels_path=/data/617/images/training_4_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="8__640"></a>
## 8       @ 640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_7_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis__8640"></a>
### vis       @ 8/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__8640"></a>
### no_aug       @ 8/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug8640"></a>
#### stitched       @ no_aug/8/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png --labels_path=/data/617/images/training_32_49/labels

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="16__640"></a>
## 16       @ 640

CUDA_VISIBLE_DEVICES=2 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_15_640_640_64_256_rot_15_345_4_flip --num_clones=1


<a id="vis__16640"></a>
### vis       @ 16/640

CUDA_VISIBLE_DEVICES=2 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__16640"></a>
### no_aug       @ 16/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug16640"></a>
#### stitched       @ no_aug/16/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png labels_path=/data/617/images/training_32_49/labels

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="16_rt__640"></a>
## 16_rt       @ 640

CUDA_VISIBLE_DEVICES=2 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=3 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_15_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="no_aug__16_rt640"></a>
### no_aug       @ 16_rt/640

CUDA_VISIBLE_DEVICES=1 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug16_rt640"></a>
#### stitched       @ no_aug/16_rt/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png labels_path=/data/617/images/training_32_49/labels

<a id="16_rt_3__640"></a>
## 16_rt_3       @ 640

CUDA_VISIBLE_DEVICES=2 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=3 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_15_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="no_aug__16_rt_3640"></a>
### no_aug       @ 16_rt_3/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug16_rt_3640"></a>
#### stitched       @ no_aug/16_rt_3/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png labels_path=/data/617/images/training_32_49/labels

<a id="24__640"></a>
## 24       @ 640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_23_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis__24640"></a>
### vis       @ 24/640

CUDA_VISIBLE_DEVICES=2 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__24640"></a>
### no_aug       @ 24/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug24640"></a>
#### stitched       @ no_aug/24/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="32__640-1"></a>
## 32       @ 640

CUDA_VISIBLE_DEVICES=1 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_31_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis__32640"></a>
### vis       @ 32/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__32640"></a>
### no_aug       @ 32/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug32640"></a>
#### stitched       @ no_aug/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1 --normalize_labels=1

<a id="yun00001__32640"></a>
### YUN00001       @ 32/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_0_8999_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=YUN00001_0_8999_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_0_8999_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png out_ext=mkv width=1920 height=1080

<a id="yun00001_3600__32640"></a>
### YUN00001_3600       @ 32/640

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600_0_3599_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=YUN00001_3600_0_3599_640_640_640_640 --vis_batch_size=25 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600_0_3599_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

<a id="4__non_aug__640"></a>
## 4__non_aug       @ 640

<a id="sel_2__4__non_aug640"></a>
### sel_2       @ 4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_2 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_2 --num_clones=1

<a id="sel_10__4__non_aug640"></a>
### sel_10       @ 4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_10 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_10 --num_clones=1

<a id="sel_100__4__non_aug640"></a>
### sel_100       @ 4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_100 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_100 --num_clones=1

<a id="sel_1000__4__non_aug640"></a>
### sel_1000       @ 4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_1000 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_1000 --num_clones=1

<a id="rt__sel_10004__non_aug640"></a>
#### rt       @ sel_1000/4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_1000_rt --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_1000 --num_clones=1

<a id="rt2__sel_10004__non_aug640"></a>
#### rt2       @ sel_1000/4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_1000_rt2 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_1000 --num_clones=1

<a id="800"></a>
# 800

<a id="build_data__800"></a>
## build_data       @ 800

<a id="50__build_data800"></a>
### 50       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

<a id="32__build_data800"></a>
### 32       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

<a id="18-test__build_data800"></a>
### 18_-_test       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

<a id="validation020_800_800_800_800__build_data800"></a>
### validation_0_20_800_800_800_800       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_800_800_800_800 --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

<a id="yun000010239_800_800_800_800__build_data800"></a>
### YUN00001_0_239_800_800_800_800       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_800_800_800_800 --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip --create_dummy_labels=1

<a id="4__build_data800"></a>
### 4       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip


<a id="50__800"></a>
## 50       @ 800

CUDA_VISIBLE_DEVICES=1 python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=800 --train_crop_size=800 --train_batch_size=2 --dataset=training_0_31_49_800_800_80_320_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_800_800_80_320_rot_15_345_4_flip/tfrecord --train_split=training_0_49_800_800_80_320_rot_15_345_4_flip --num_clones=1


<a id="eval__50800"></a>
### eval       @ 50/800

CUDA_VISIBLE_DEVICES=1 python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=800 --eval_crop_size=800 --dataset="training_0_31_49_800_800_80_320_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49 --eval_logdir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_800_800_80_320_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_800_800_80_320_rot_15_345_4_flip --eval_batch_size=5

<a id="vis__50800"></a>
### vis       @ 50/800

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=800 --vis_crop_size=800 --dataset="training_0_31_49_800_800_80_320_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49/vis/training_32_49_800_800_80_320_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_800_800_80_320_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_800_800_80_320_rot_15_345_4_flip --vis_batch_size=10 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49/vis/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="stitching__vis50800"></a>
#### stitching       @ vis/50/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/training_32_49_640_640_64_256_rot_15_345_4_flip patch_height=640 patch_width=640 start_id=32 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=1

<a id="validation__50800"></a>
### validation       @ 50/800

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="stitching__validation50800"></a>
#### stitching       @ validation/50/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640/raw patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

<a id="vis__validation50800"></a>
#### vis       @ validation/50/800

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="4__800"></a>
## 4       @ 800

CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=800 --train_crop_size=800 --train_batch_size=2 --dataset=training_0_31_49_800_800_80_320_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_800_800_80_320_rot_15_345_4_flip/tfrecord --train_split=training_0_3_800_800_80_320_rot_15_345_4_flip --num_clones=1



