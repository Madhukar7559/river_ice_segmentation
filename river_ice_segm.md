https://github.com/abhineet123/617_w18_proj_code/blob/master/proj.md

<!--ts-->
   * [320x640](#320x640)
   * [256](#256)
      * [all](#all)
         * [ppt](#ppt)
         * [rotation and flipping](#rotation-and-flipping)
         * [merging](#merging)
      * [0-31](#0-31)
         * [merging](#merging-1)
      * [32-49](#32-49)
         * [merging](#merging-2)
      * [batch all](#batch-all)
      * [validation](#validation)
         * [stitching](#stitching)
      * [videos](#videos)
         * [stitching](#stitching-1)
   * [384](#384)
      * [40/160](#40160)
      * [25/100](#25100)
      * [validation](#validation-1)
         * [stitching](#stitching-2)
      * [videos](#videos-1)
         * [stitching](#stitching-3)
      * [vis](#vis)
         * [unet](#unet)
            * [hml](#hml)
            * [weird](#weird)
   * [512](#512)
      * [40/160](#40160-1)
      * [25/100](#25100-1)
      * [validation](#validation-2)
         * [stitching](#stitching-4)
      * [videos](#videos-2)
         * [stitching](#stitching-5)
   * [640](#640)
      * [64/256](#64256)
      * [25/100](#25100-2)
      * [validation](#validation-3)
         * [stitching](#stitching-6)
      * [videos](#videos-3)
         * [stitching](#stitching-7)
   * [800](#800)
      * [80/320](#80320)
      * [25/100](#25100-3)
      * [video](#video)
   * [1000](#1000)
      * [100/400](#100400)
   * [stitch multiple results](#stitch-multiple-results)

<!-- Added by: abhineet, at: 2018-04-26T21:34-06:00 -->

<!--te-->


# videoToImgSeq

## 1920x1080

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=0.50 dst_dir=/data/617/images/YUN00001_1920x1080/images

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=0.50 dst_dir=/data/617/images/YUN00002_1920x1080/images

## 4k

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=-1 resize_factor=1 dst_dir=/data/617/images/YUN00001/images

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=1 dst_dir=/data/617/images/YUN00001_1800/images

### YUN00001_3600

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=3600 resize_factor=1 dst_dir=/data/617/images/YUN00001_3600/images

### YUN00001_3600 - win

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=3600 resize_factor=1 dst_dir=E:\Datasets\617\images\YUN00001_3600\images

### 20160121_YUN00002_2000

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=2000 dst_dir=/data/617/images/20160121_YUN00002_2000/images 

### 20161201_YUN00002_1800

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161201 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161201_YUN00002_1800/images

### 20160122_YUN00002_700_2500

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160122 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=700 dst_dir=/data/617/images/20160122_YUN00002_700_2500/images 

### 20160122_YUN00020_2000_3800

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160122 seq_name=YUN00020 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=2000 dst_dir=/data/617/images/20160122_YUN00020_2000_3800/images

### 20160122_YUN00020_2000_3800 - win_pc

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20160122 seq_name=YUN00020 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=2000 dst_dir=P:\Datasets\617\images\20160122_YUN00020_2000_300\images

### 20161203_Deployment_1_YUN00001_900_2700

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=900 dst_dir=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images

### 20161203_Deployment_1_YUN00001_900_1200 - win_pc

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=2000 dst_dir=P:\Datasets\617\images\20161203_Deployment_1_YUN00001_900_1200\images

### 20161203_Deployment_1_YUN00002_1800

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161203 seq_name=Deployment_1_YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161203_Deployment_1_YUN00002_1800/images


### 20170114_YUN00005_1800

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20170114 seq_name=YUN00005 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20170114_YUN00005_1800/images 

# 320x640

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=320 patch_width=640 min_stride=25 max_stride=100

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=320 patch_width=640 min_stride=100 max_stride=200

# 256


## all

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=100 max_stride=200

### ppt

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1 start_id=2 end_id=2

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1 start_id=2 end_id=2 enable_rot=1 min_rot=15 max_rot=90


### rotation and flipping

python3 subPatchDataset.py db_root_dir=/home/abhineet/N/Datasets/617 seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=20 max_rot=75

python3 subPatchDataset.py db_root_dir=/home/abhineet/N/Datasets/617 seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=20 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=100 max_stride=200 enable_rot=1 min_rot=90 max_rot=180 enable_flip=1

### merging

python3 mergeDatasets.py training_256_256_100_200_flip training_256_256_100_200_rot_90_180_flip 

## 0-31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=0 end_id=31

### merging
python3 mergeDatasets.py training_0_31_256_256_25_100_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_126_235_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_15_125_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_236_345_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

## 32-49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=32 end_id=49

### merging

python3 mergeDatasets.py training_32_49_256_256_25_100_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_126_235_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_15_125_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_236_345_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

## batch all

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

## validation

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=1 end_id=1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

### stitching

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_1_1_256_256_256_256 patch_height=256 patch_width=256 start_id=1 end_id=1 patch_seq_type=images show_img=1 stacked=1 method=1 resize_factor=0.5

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_1_1_256_256_256_256 patch_height=256 patch_width=256 start_id=1 end_id=1 patch_seq_type=labels_deeplab_xception show_img=1 stacked=1 method=1 resize_factor=0.5

## videos

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

### stitching

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1 

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1


# 384

## 40/160

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49



## 25/100

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

## validation

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=20


### stitching

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_384_384_384_384 patch_height=384 patch_width=384 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

## videos

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

### stitching

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_384_384_384_384 patch_height=384 patch_width=384 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1

## vis

### unet

#### hml

python3 visDataset.py --images_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/images --labels_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/vis --n_classes=3 --start_id=0 --end_id=-1

python3 visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

#### weird

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1


python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_50_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=50 patch_seq_type=images show_img=1 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=vgg_unet2_max_val_acc_validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0 method=1
python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=fcn32_max_mean_acc_validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0 method=1


# 512

## 40/160


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

## 25/100


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

## validation

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=20

### stitching

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_512_512_512_512 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

## videos

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

### stitching

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_512_512_512_512 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1

# 640

## 64/256

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

## non_aug

### 0 - 3

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

#### sel-2

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=2

#### sel-10

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=10

#### sel-100

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=100

#### sel-1000

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=1000

#### sel-5000

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=5000

### 32 -49

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

### 0 - 49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=49 img_ext=tif

### 4 - 49

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

### entire image

#### 32-49

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

#### 4-49

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

### ablation

#### 0 - 3

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=3

#### sel-2

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=2

#### sel-2

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=2

#### sel-10

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=10

#### sel-100

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=100

#### sel-1000

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=1000

#### sel-5000

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=5000

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=7

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=15

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=23

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

## 25/100

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49


python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

## validation

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=20

### stitching

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

## videos

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_3600 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700_2500 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00020_2000_3800 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00001_900_2700 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00002_1800 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


### stitching

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1


# 800

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

## 80/320

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

## non_aug

### 0 - 3

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

### 32 - 49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

### 0 - 49

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=49 img_ext=tif

### 4 - 49

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

### entire image

#### 32-49

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

#### 4-49

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

### ablation

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=3

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=7

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=15

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=23

## 25/100

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

## video

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_3600 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700_2500 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00020_2000_3800 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00001_900_2700 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00002_1800 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg




python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_2000 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20161201_YUN00002 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20170114_YUN00005 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

# 1000

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0


## 100/400

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

## video

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_2000 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20161201_YUN00002 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20170114_YUN00005 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

### 1920x1080

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_1920x1080 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_1920x1080 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


# stitch multiple results

python3 stitchMultipleResults.py --seg_root_dir=/home/abhineet/H/UofA/617/Project/presentation --images_path=/data/617/images/validation/images --save_path=/home/abhineet/H/UofA/617/Project/presentation/stitched --show_img=1 --resize_factor=0.25

# svm

## 4

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_4_1  --save_path=svm\svm_1_4_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_4_2  --save_path=svm\svm_1_4_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

## 8

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_8_1  --save_path=svm\svm_1_8_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_8_2  --save_path=svm\svm_1_8_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1


## 16

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_16_1  --save_path=svm\svm_1_16_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_16_2  --save_path=svm\svm_1_16_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

## 24

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_24_1  --save_path=svm\svm_1_24_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_24_2  --save_path=svm\svm_1_24_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

## 32

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_32_1  --save_path=svm\svm_1_32_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_32_2  --save_path=svm\svm_1_32_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1










