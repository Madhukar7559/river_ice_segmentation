https://github.com/abhineet123/617_w18_proj_code/blob/master/proj.md

<!-- MarkdownTOC -->

- [plotIceConcentration](#ploticeconcentration)
   - [labeled       @ plotIceConcentration](#labeled__ploticeconcentration)
      - [deeplab       @ labeled/plotIceConcentration](#deeplab__labeledploticeconcentration)
         - [anchor       @ deeplab/labeled/plotIceConcentration](#anchor__deeplablabeledploticeconcentration)
         - [frazil       @ deeplab/labeled/plotIceConcentration](#frazil__deeplablabeledploticeconcentration)
      - [densenet       @ labeled/plotIceConcentration](#densenet__labeledploticeconcentration)
         - [anchor       @ densenet/labeled/plotIceConcentration](#anchor__densenetlabeledploticeconcentration)
         - [frazil       @ densenet/labeled/plotIceConcentration](#frazil__densenetlabeledploticeconcentration)
      - [svm       @ labeled/plotIceConcentration](#svm__labeledploticeconcentration)
         - [anchor       @ svm/labeled/plotIceConcentration](#anchor__svmlabeledploticeconcentration)
         - [frazil       @ svm/labeled/plotIceConcentration](#frazil__svmlabeledploticeconcentration)
      - [svm_and_deeplab       @ labeled/plotIceConcentration](#svm_and_deeplab__labeledploticeconcentration)
   - [video       @ plotIceConcentration](#video__ploticeconcentration)
      - [20160122_YUN00002_700_2500       @ video/plotIceConcentration](#20160122_yun00002_700_2500__videoploticeconcentration)
         - [combined       @ 20160122_YUN00002_700_2500/video/plotIceConcentration](#combined__20160122_yun00002_700_2500videoploticeconcentration)
            - [plot_changed_seg_count       @ combined/20160122_YUN00002_700_2500/video/plotIceConcentration](#plot_changed_seg_count__combined20160122_yun00002_700_2500videoploticeconcentration)
         - [frazil       @ 20160122_YUN00002_700_2500/video/plotIceConcentration](#frazil__20160122_yun00002_700_2500videoploticeconcentration)
         - [anchor       @ 20160122_YUN00002_700_2500/video/plotIceConcentration](#anchor__20160122_yun00002_700_2500videoploticeconcentration)
      - [20160122_YUN00020_2000_3800       @ video/plotIceConcentration](#20160122_yun00020_2000_3800__videoploticeconcentration)
         - [combined       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration](#combined__20160122_yun00020_2000_3800videoploticeconcentration)
            - [svm       @ combined/20160122_YUN00020_2000_3800/video/plotIceConcentration](#svm__combined20160122_yun00020_2000_3800videoploticeconcentration)
         - [frazil       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration](#frazil__20160122_yun00020_2000_3800videoploticeconcentration)
            - [svm       @ frazil/20160122_YUN00020_2000_3800/video/plotIceConcentration](#svm__frazil20160122_yun00020_2000_3800videoploticeconcentration)
         - [anchor       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration](#anchor__20160122_yun00020_2000_3800videoploticeconcentration)
            - [svm       @ anchor/20160122_YUN00020_2000_3800/video/plotIceConcentration](#svm__anchor20160122_yun00020_2000_3800videoploticeconcentration)
      - [20161203_Deployment_1_YUN00002_1800       @ video/plotIceConcentration](#20161203_deployment1yun00002_1800__videoploticeconcentration)
         - [combined       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration](#combined__20161203_deployment1yun00002_1800videoploticeconcentration)
         - [frazil       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration](#frazil__20161203_deployment1yun00002_1800videoploticeconcentration)
         - [anchor       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration](#anchor__20161203_deployment1yun00002_1800videoploticeconcentration)
      - [20161203_Deployment_1_YUN00001_900_2700       @ video/plotIceConcentration](#20161203_deployment1yun00001_900_2700__videoploticeconcentration)
         - [combined       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#combined__20161203_deployment1yun00001_900_2700videoploticeconcentration)
            - [svm       @ combined/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#svm__combined20161203_deployment1yun00001_900_2700videoploticeconcentration)
         - [frazil       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#frazil__20161203_deployment1yun00001_900_2700videoploticeconcentration)
            - [svm       @ frazil/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#svm__frazil20161203_deployment1yun00001_900_2700videoploticeconcentration)
         - [anchor       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#anchor__20161203_deployment1yun00001_900_2700videoploticeconcentration)
            - [svm       @ anchor/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#svm__anchor20161203_deployment1yun00001_900_2700videoploticeconcentration)
      - [YUN00001_3600       @ video/plotIceConcentration](#yun00001_3600__videoploticeconcentration)
         - [combined       @ YUN00001_3600/video/plotIceConcentration](#combined__yun00001_3600videoploticeconcentration)
            - [svm       @ combined/YUN00001_3600/video/plotIceConcentration](#svm__combinedyun00001_3600videoploticeconcentration)
         - [frazil       @ YUN00001_3600/video/plotIceConcentration](#frazil__yun00001_3600videoploticeconcentration)
            - [svm       @ frazil/YUN00001_3600/video/plotIceConcentration](#svm__frazilyun00001_3600videoploticeconcentration)
         - [anchor       @ YUN00001_3600/video/plotIceConcentration](#anchor__yun00001_3600videoploticeconcentration)
            - [svm       @ anchor/YUN00001_3600/video/plotIceConcentration](#svm__anchoryun00001_3600videoploticeconcentration)
- [videoToImgSeq](#videotoimgseq)
   - [1920x1080       @ videoToImgSeq](#1920x1080__videotoimgseq)
   - [4k       @ videoToImgSeq](#4k__videotoimgseq)
      - [YUN00001_3600       @ 4k/videoToImgSeq](#yun00001_3600__4kvideotoimgseq)
      - [YUN00001_3600_-_win       @ 4k/videoToImgSeq](#yun00001_3600-win__4kvideotoimgseq)
      - [20160121_YUN00002_2000       @ 4k/videoToImgSeq](#20160121_yun00002_2000__4kvideotoimgseq)
      - [20161201_YUN00002_1800       @ 4k/videoToImgSeq](#20161201_yun00002_1800__4kvideotoimgseq)
      - [20160122_YUN00002_700_2500       @ 4k/videoToImgSeq](#20160122_yun00002_700_2500__4kvideotoimgseq)
      - [20160122_YUN00020_2000_3800       @ 4k/videoToImgSeq](#20160122_yun00020_2000_3800__4kvideotoimgseq)
      - [20160122_YUN00020_2000_3800__win_pc       @ 4k/videoToImgSeq](#20160122_yun00020_2000_3800__win_pc__4kvideotoimgseq)
      - [20161203_Deployment_1_YUN00001_900_2700       @ 4k/videoToImgSeq](#20161203_deployment1yun00001_900_2700__4kvideotoimgseq)
      - [20161203_Deployment_1_YUN00001_900_1200_win_pc       @ 4k/videoToImgSeq](#20161203_deployment1yun00001_900_1200_win_pc__4kvideotoimgseq)
      - [20161203_Deployment_1_YUN00001_2000_2300__win_pc       @ 4k/videoToImgSeq](#20161203_deployment1yun00001_2000_2300__win_pc__4kvideotoimgseq)
      - [20161203_Deployment_1_YUN00002_1800       @ 4k/videoToImgSeq](#20161203_deployment1yun00002_1800__4kvideotoimgseq)
      - [20170114_YUN00005_1800       @ 4k/videoToImgSeq](#20170114_yun00005_1800__4kvideotoimgseq)
- [320x640](#320x640)
- [256](#256)
   - [all       @ 256](#all__256)
      - [ppt       @ all/256](#ppt__all256)
      - [rotation_and_flipping       @ all/256](#rotation_and_flipping__all256)
      - [merging       @ all/256](#merging__all256)
   - [0-31       @ 256](#0-31__256)
      - [merging       @ 0-31/256](#merging__0-31256)
   - [32-49       @ 256](#32-49__256)
      - [merging       @ 32-49/256](#merging__32-49256)
   - [batch_all       @ 256](#batch_all__256)
   - [validation       @ 256](#validation__256)
      - [stitching       @ validation/256](#stitching__validation256)
   - [videos       @ 256](#videos__256)
      - [stitching       @ videos/256](#stitching__videos256)
- [384](#384)
   - [40/160       @ 384](#40160__384)
   - [25/100       @ 384](#25100__384)
   - [validation       @ 384](#validation__384)
      - [stitching       @ validation/384](#stitching__validation384)
   - [videos       @ 384](#videos__384)
      - [stitching       @ videos/384](#stitching__videos384)
   - [vis       @ 384](#vis__384)
      - [unet       @ vis/384](#unet__vis384)
         - [hml       @ unet/vis/384](#hml__unetvis384)
         - [weird       @ unet/vis/384](#weird__unetvis384)
- [512](#512)
   - [40/160       @ 512](#40160__512)
   - [25/100       @ 512](#25100__512)
   - [validation       @ 512](#validation__512)
      - [stitching       @ validation/512](#stitching__validation512)
   - [videos       @ 512](#videos__512)
      - [stitching       @ videos/512](#stitching__videos512)
- [640](#640)
   - [64/256       @ 640](#64256__640)
   - [non_aug       @ 640](#non_aug__640)
      - [0_-_3       @ non_aug/640](#0-3__non_aug640)
         - [sel-2       @ 0_-_3/non_aug/640](#sel-2__0-3non_aug640)
         - [sel-10       @ 0_-_3/non_aug/640](#sel-10__0-3non_aug640)
         - [sel-100       @ 0_-_3/non_aug/640](#sel-100__0-3non_aug640)
         - [sel-1000       @ 0_-_3/non_aug/640](#sel-1000__0-3non_aug640)
         - [sel-5000       @ 0_-_3/non_aug/640](#sel-5000__0-3non_aug640)
      - [32_-49       @ non_aug/640](#32_-49__non_aug640)
      - [0_-_49       @ non_aug/640](#0-49__non_aug640)
      - [4_-_49       @ non_aug/640](#4-49__non_aug640)
      - [entire_image       @ non_aug/640](#entire_image__non_aug640)
         - [32-49       @ entire_image/non_aug/640](#32-49__entire_imagenon_aug640)
         - [4-49       @ entire_image/non_aug/640](#4-49__entire_imagenon_aug640)
      - [ablation       @ non_aug/640](#ablation__non_aug640)
         - [0_-_3       @ ablation/non_aug/640](#0-3__ablationnon_aug640)
         - [sel-2       @ ablation/non_aug/640](#sel-2__ablationnon_aug640)
         - [sel-2       @ ablation/non_aug/640](#sel-2__ablationnon_aug640-1)
         - [sel-10       @ ablation/non_aug/640](#sel-10__ablationnon_aug640)
         - [sel-100       @ ablation/non_aug/640](#sel-100__ablationnon_aug640)
         - [sel-1000       @ ablation/non_aug/640](#sel-1000__ablationnon_aug640)
         - [sel-5000       @ ablation/non_aug/640](#sel-5000__ablationnon_aug640)
   - [25/100       @ 640](#25100__640)
   - [validation       @ 640](#validation__640)
      - [stitching       @ validation/640](#stitching__validation640)
   - [videos       @ 640](#videos__640)
      - [stitching       @ videos/640](#stitching__videos640)
- [800](#800)
   - [80/320       @ 800](#80320__800)
   - [non_aug       @ 800](#non_aug__800)
      - [0_-_3       @ non_aug/800](#0-3__non_aug800)
      - [32_-_49       @ non_aug/800](#32-49__non_aug800)
      - [0_-_49       @ non_aug/800](#0-49__non_aug800)
      - [4_-_49       @ non_aug/800](#4-49__non_aug800)
      - [entire_image       @ non_aug/800](#entire_image__non_aug800)
         - [32-49       @ entire_image/non_aug/800](#32-49__entire_imagenon_aug800)
         - [4-49       @ entire_image/non_aug/800](#4-49__entire_imagenon_aug800)
      - [ablation       @ non_aug/800](#ablation__non_aug800)
   - [25/100       @ 800](#25100__800)
   - [video       @ 800](#video__800)
- [1000](#1000)
   - [100/400       @ 1000](#100400__1000)
   - [video       @ 1000](#video__1000)
      - [1920x1080       @ video/1000](#1920x1080__video1000)
- [stitch multiple results](#stitch_multiple_results)
- [svm](#svm)
   - [4       @ svm](#4__svm)
   - [8       @ svm](#8__svm)
   - [16       @ svm](#16__svm)
   - [24       @ svm](#24__svm)
   - [32       @ svm](#32__svm)

<!-- /MarkdownTOC -->

<a id="ploticeconcentration"></a>
# plotIceConcentration

<a id="labeled__ploticeconcentration"></a>
## labeled       @ plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training/images --labels_path=/data/617/images/training/labels --images_ext=tif --labels_ext=tif --n_classes=3

<a id="deeplab__labeledploticeconcentration"></a>
### deeplab       @ labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\log\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab

<a id="anchor__deeplablabeledploticeconcentration"></a>
#### anchor       @ deeplab/labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\log\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab --ice_type=1

<a id="frazil__deeplablabeledploticeconcentration"></a>
#### frazil       @ deeplab/labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\log\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab --ice_type=2

<a id="densenet__labeledploticeconcentration"></a>
### densenet       @ labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\log\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet

<a id="anchor__densenetlabeledploticeconcentration"></a>
#### anchor       @ densenet/labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\log\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet --ice_type=1

<a id="frazil__densenetlabeledploticeconcentration"></a>
#### frazil       @ densenet/labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\log\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet --ice_type=2


<a id="svm__labeledploticeconcentration"></a>
### svm       @ labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_1 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_1

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2


<a id="anchor__svmlabeledploticeconcentration"></a>
#### anchor       @ svm/labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2 --ice_type=1

<a id="frazil__svmlabeledploticeconcentration"></a>
#### frazil       @ svm/labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2 --ice_type=2

<a id="svm_and_deeplab__labeledploticeconcentration"></a>
### svm_and_deeplab       @ labeled/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2,H:\UofA\617\Project\617_proj_code\log\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab --ice_type=1

<a id="video__ploticeconcentration"></a>
## video       @ plotIceConcentration

<a id="20160122_yun00002_700_2500__videoploticeconcentration"></a>
### 20160122_YUN00002_700_2500       @ video/plotIceConcentration

<a id="combined__20160122_yun00002_700_2500videoploticeconcentration"></a>
#### combined       @ 20160122_YUN00002_700_2500/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="plot_changed_seg_count__combined20160122_yun00002_700_2500videoploticeconcentration"></a>
##### plot_changed_seg_count       @ combined/20160122_YUN00002_700_2500/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 --plot_changed_seg_count=1

<a id="frazil__20160122_yun00002_700_2500videoploticeconcentration"></a>
#### frazil       @ 20160122_YUN00002_700_2500/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="anchor__20160122_yun00002_700_2500videoploticeconcentration"></a>
#### anchor       @ 20160122_YUN00002_700_2500/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="20160122_yun00020_2000_3800__videoploticeconcentration"></a>
### 20160122_YUN00020_2000_3800       @ video/plotIceConcentration

<a id="combined__20160122_yun00020_2000_3800videoploticeconcentration"></a>
#### combined       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 

<a id="svm__combined20160122_yun00020_2000_3800videoploticeconcentration"></a>
##### svm       @ combined/20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299


<a id="frazil__20160122_yun00020_2000_3800videoploticeconcentration"></a>
#### frazil       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720

<a id="svm__frazil20160122_yun00020_2000_3800videoploticeconcentration"></a>
##### svm       @ frazil/20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299

<a id="anchor__20160122_yun00020_2000_3800videoploticeconcentration"></a>
#### anchor       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 

<a id="svm__anchor20160122_yun00020_2000_3800videoploticeconcentration"></a>
##### svm       @ anchor/20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299


<a id="20161203_deployment1yun00002_1800__videoploticeconcentration"></a>
### 20161203_Deployment_1_YUN00002_1800       @ video/plotIceConcentration

<a id="combined__20161203_deployment1yun00002_1800videoploticeconcentration"></a>
#### combined       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 

<a id="frazil__20161203_deployment1yun00002_1800videoploticeconcentration"></a>
#### frazil       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 

<a id="anchor__20161203_deployment1yun00002_1800videoploticeconcentration"></a>
#### anchor       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 

<a id="20161203_deployment1yun00001_900_2700__videoploticeconcentration"></a>
### 20161203_Deployment_1_YUN00001_900_2700       @ video/plotIceConcentration

<a id="combined__20161203_deployment1yun00001_900_2700videoploticeconcentration"></a>
#### combined       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm__combined20161203_deployment1yun00001_900_2700videoploticeconcentration"></a>
##### svm       @ combined/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299



<a id="frazil__20161203_deployment1yun00001_900_2700videoploticeconcentration"></a>
#### frazil       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm__frazil20161203_deployment1yun00001_900_2700videoploticeconcentration"></a>
##### svm       @ frazil/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299


<a id="anchor__20161203_deployment1yun00001_900_2700videoploticeconcentration"></a>
#### anchor       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm__anchor20161203_deployment1yun00001_900_2700videoploticeconcentration"></a>
##### svm       @ anchor/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299


<a id="yun00001_3600__videoploticeconcentration"></a>
### YUN00001_3600       @ video/plotIceConcentration

<a id="combined__yun00001_3600videoploticeconcentration"></a>
#### combined       @ YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/YUN00001_3600  --out_size=1920x720 

<a id="svm__combinedyun00001_3600videoploticeconcentration"></a>
##### svm       @ combined/YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899

<a id="frazil__yun00001_3600videoploticeconcentration"></a>
#### frazil       @ YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/YUN00001_3600  --out_size=1920x720 

<a id="svm__frazilyun00001_3600videoploticeconcentration"></a>
##### svm       @ frazil/YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899


<a id="anchor__yun00001_3600videoploticeconcentration"></a>
#### anchor       @ YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/YUN00001_3600  --out_size=1920x720 

<a id="svm__anchoryun00001_3600videoploticeconcentration"></a>
##### svm       @ anchor/YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899







<a id="videotoimgseq"></a>
# videoToImgSeq

<a id="1920x1080__videotoimgseq"></a>
## 1920x1080       @ videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=0.50 dst_dir=/data/617/images/YUN00001_1920x1080/images

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_fa ctor=0.50 dst_dir=/data/617/images/YUN00002_1920x1080/images

<a id="4k__videotoimgseq"></a>
## 4k       @ videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=-1 resize_factor=1 dst_dir=/data/617/images/YUN00001/images

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=1 dst_dir=/data/617/images/YUN00001_1800/images

<a id="yun00001_3600__4kvideotoimgseq"></a>
### YUN00001_3600       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=3600 resize_factor=1 dst_dir=/data/617/images/YUN00001_3600/images

<a id="yun00001_3600-win__4kvideotoimgseq"></a>
### YUN00001_3600_-_win       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=3600 resize_factor=1 dst_dir=E:\Datasets\617\images\YUN00001_3600\images

<a id="20160121_yun00002_2000__4kvideotoimgseq"></a>
### 20160121_YUN00002_2000       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=2000 dst_dir=/data/617/images/20160121_YUN00002_2000/images 

<a id="20161201_yun00002_1800__4kvideotoimgseq"></a>
### 20161201_YUN00002_1800       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161201 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161201_YUN00002_1800/images

<a id="20160122_yun00002_700_2500__4kvideotoimgseq"></a>
### 20160122_YUN00002_700_2500       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160122 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=700 dst_dir=/data/617/images/20160122_YUN00002_700_2500/images 

<a id="20160122_yun00020_2000_3800__4kvideotoimgseq"></a>
### 20160122_YUN00020_2000_3800       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160122 seq_name=YUN00020 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=2000 dst_dir=/data/617/images/20160122_YUN00020_2000_3800/images

<a id="20160122_yun00020_2000_3800__win_pc__4kvideotoimgseq"></a>
### 20160122_YUN00020_2000_3800__win_pc       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20160122 seq_name=YUN00020 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=2000 dst_dir=P:\Datasets\617\images\20160122_YUN00020_2000_300\images

<a id="20161203_deployment1yun00001_900_2700__4kvideotoimgseq"></a>
### 20161203_Deployment_1_YUN00001_900_2700       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=900 dst_dir=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images

<a id="20161203_deployment1yun00001_900_1200_win_pc__4kvideotoimgseq"></a>
### 20161203_Deployment_1_YUN00001_900_1200_win_pc       @ 4k/videoToImgSeq

__corrected__
python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=900 dst_dir=P:\Datasets\617\images\20161203_Deployment_1_YUN00001_900_1200\images

<a id="20161203_deployment1yun00001_2000_2300__win_pc__4kvideotoimgseq"></a>
### 20161203_Deployment_1_YUN00001_2000_2300__win_pc       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=2000 dst_dir=E:\Datasets\617\images\20161203_Deployment_1_YUN00001_2000_2300\images

<a id="20161203_deployment1yun00002_1800__4kvideotoimgseq"></a>
### 20161203_Deployment_1_YUN00002_1800       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161203 seq_name=Deployment_1_YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161203_Deployment_1_YUN00002_1800/images


<a id="20170114_yun00005_1800__4kvideotoimgseq"></a>
### 20170114_YUN00005_1800       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20170114 seq_name=YUN00005 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20170114_YUN00005_1800/images 

<a id="320x640"></a>
# 320x640

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=320 patch_width=640 min_stride=25 max_stride=100

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=320 patch_width=640 min_stride=100 max_stride=200

<a id="256"></a>
# 256


<a id="all__256"></a>
## all       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=100 max_stride=200

<a id="ppt__all256"></a>
### ppt       @ all/256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1 start_id=2 end_id=2

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1 start_id=2 end_id=2 enable_rot=1 min_rot=15 max_rot=90


<a id="rotation_and_flipping__all256"></a>
### rotation_and_flipping       @ all/256

python3 subPatchDataset.py db_root_dir=/home/abhineet/N/Datasets/617 seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=20 max_rot=75

python3 subPatchDataset.py db_root_dir=/home/abhineet/N/Datasets/617 seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=20 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=100 max_stride=200 enable_rot=1 min_rot=90 max_rot=180 enable_flip=1

<a id="merging__all256"></a>
### merging       @ all/256

python3 mergeDatasets.py training_256_256_100_200_flip training_256_256_100_200_rot_90_180_flip 

<a id="0-31__256"></a>
## 0-31       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=0 end_id=31

<a id="merging__0-31256"></a>
### merging       @ 0-31/256
python3 mergeDatasets.py training_0_31_256_256_25_100_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_126_235_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_15_125_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_236_345_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

<a id="32-49__256"></a>
## 32-49       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=32 end_id=49

<a id="merging__32-49256"></a>
### merging       @ 32-49/256

python3 mergeDatasets.py training_32_49_256_256_25_100_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_126_235_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_15_125_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_236_345_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

<a id="batch_all__256"></a>
## batch_all       @ 256

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

<a id="validation__256"></a>
## validation       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=1 end_id=1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

<a id="stitching__validation256"></a>
### stitching       @ validation/256

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_1_1_256_256_256_256 patch_height=256 patch_width=256 start_id=1 end_id=1 patch_seq_type=images show_img=1 stacked=1 method=1 resize_factor=0.5

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_1_1_256_256_256_256 patch_height=256 patch_width=256 start_id=1 end_id=1 patch_seq_type=labels_deeplab_xception show_img=1 stacked=1 method=1 resize_factor=0.5

<a id="videos__256"></a>
## videos       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching__videos256"></a>
### stitching       @ videos/256

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1 

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1


<a id="384"></a>
# 384

<a id="40160__384"></a>
## 40/160       @ 384

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49



<a id="25100__384"></a>
## 25/100       @ 384

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

<a id="validation__384"></a>
## validation       @ 384

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=20


<a id="stitching__validation384"></a>
### stitching       @ validation/384

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_384_384_384_384 patch_height=384 patch_width=384 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos__384"></a>
## videos       @ 384

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching__videos384"></a>
### stitching       @ videos/384

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_384_384_384_384 patch_height=384 patch_width=384 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1

<a id="vis__384"></a>
## vis       @ 384

<a id="unet__vis384"></a>
### unet       @ vis/384

<a id="hml__unetvis384"></a>
#### hml       @ unet/vis/384

python3 visDataset.py --images_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/images --labels_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/vis --n_classes=3 --start_id=0 --end_id=-1

python3 visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="weird__unetvis384"></a>
#### weird       @ unet/vis/384

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1


python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_50_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=50 patch_seq_type=images show_img=1 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=vgg_unet2_max_val_acc_validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0 method=1
python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=fcn32_max_mean_acc_validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0 method=1


<a id="512"></a>
# 512

<a id="40160__512"></a>
## 40/160       @ 512


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="25100__512"></a>
## 25/100       @ 512


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="validation__512"></a>
## validation       @ 512

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=20

<a id="stitching__validation512"></a>
### stitching       @ validation/512

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_512_512_512_512 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos__512"></a>
## videos       @ 512

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching__videos512"></a>
### stitching       @ videos/512

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_512_512_512_512 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1

<a id="640"></a>
# 640

<a id="64256__640"></a>
## 64/256       @ 640

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="non_aug__640"></a>
## non_aug       @ 640

<a id="0-3__non_aug640"></a>
### 0_-_3       @ non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

<a id="sel-2__0-3non_aug640"></a>
#### sel-2       @ 0_-_3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=2

<a id="sel-10__0-3non_aug640"></a>
#### sel-10       @ 0_-_3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=10

<a id="sel-100__0-3non_aug640"></a>
#### sel-100       @ 0_-_3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=100

<a id="sel-1000__0-3non_aug640"></a>
#### sel-1000       @ 0_-_3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=1000

<a id="sel-5000__0-3non_aug640"></a>
#### sel-5000       @ 0_-_3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=5000

<a id="32_-49__non_aug640"></a>
### 32_-49       @ non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="0-49__non_aug640"></a>
### 0_-_49       @ non_aug/640

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=49 img_ext=tif

<a id="4-49__non_aug640"></a>
### 4_-_49       @ non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="entire_image__non_aug640"></a>
### entire_image       @ non_aug/640

<a id="32-49__entire_imagenon_aug640"></a>
#### 32-49       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="4-49__entire_imagenon_aug640"></a>
#### 4-49       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="ablation__non_aug640"></a>
### ablation       @ non_aug/640

<a id="0-3__ablationnon_aug640"></a>
#### 0_-_3       @ ablation/non_aug/640

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=3

<a id="sel-2__ablationnon_aug640"></a>
#### sel-2       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=2

<a id="sel-2__ablationnon_aug640-1"></a>
#### sel-2       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=2

<a id="sel-10__ablationnon_aug640"></a>
#### sel-10       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=10

<a id="sel-100__ablationnon_aug640"></a>
#### sel-100       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=100

<a id="sel-1000__ablationnon_aug640"></a>
#### sel-1000       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=1000

<a id="sel-5000__ablationnon_aug640"></a>
#### sel-5000       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=5000

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=7

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=15

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=23

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

<a id="25100__640"></a>
## 25/100       @ 640

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49


python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

<a id="validation__640"></a>
## validation       @ 640

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=20

<a id="stitching__validation640"></a>
### stitching       @ validation/640

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos__640"></a>
## videos       @ 640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_3600 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700_2500 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00020_2000_3800 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00001_900_2700 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00002_1800 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


<a id="stitching__videos640"></a>
### stitching       @ videos/640

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1


<a id="800"></a>
# 800

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

<a id="80320__800"></a>
## 80/320       @ 800

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="non_aug__800"></a>
## non_aug       @ 800

<a id="0-3__non_aug800"></a>
### 0_-_3       @ non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

<a id="32-49__non_aug800"></a>
### 32_-_49       @ non_aug/800

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="0-49__non_aug800"></a>
### 0_-_49       @ non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=49 img_ext=tif

<a id="4-49__non_aug800"></a>
### 4_-_49       @ non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="entire_image__non_aug800"></a>
### entire_image       @ non_aug/800

<a id="32-49__entire_imagenon_aug800"></a>
#### 32-49       @ entire_image/non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="4-49__entire_imagenon_aug800"></a>
#### 4-49       @ entire_image/non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="ablation__non_aug800"></a>
### ablation       @ non_aug/800

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=3

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=7

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=15

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=23

<a id="25100__800"></a>
## 25/100       @ 800

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="video__800"></a>
## video       @ 800

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

<a id="1000"></a>
# 1000

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0


<a id="100400__1000"></a>
## 100/400       @ 1000

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="video__1000"></a>
## video       @ 1000

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_2000 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20161201_YUN00002 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20170114_YUN00005 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="1920x1080__video1000"></a>
### 1920x1080       @ video/1000

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_1920x1080 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_1920x1080 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


<a id="stitch_multiple_results"></a>
# stitch multiple results

python3 stitchMultipleResults.py --seg_root_dir=/home/abhineet/H/UofA/617/Project/presentation --images_path=/data/617/images/validation/images --save_path=/home/abhineet/H/UofA/617/Project/presentation/stitched --show_img=1 --resize_factor=0.25

<a id="svm"></a>
# svm

<a id="4__svm"></a>
## 4       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_4_1  --save_path=svm\svm_1_4_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_4_2  --save_path=svm\svm_1_4_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

<a id="8__svm"></a>
## 8       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_8_1  --save_path=svm\svm_1_8_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_8_2  --save_path=svm\svm_1_8_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1


<a id="16__svm"></a>
## 16       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_16_1  --save_path=svm\svm_1_16_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_16_2  --save_path=svm\svm_1_16_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

<a id="24__svm"></a>
## 24       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_24_1  --save_path=svm\svm_1_24_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_24_2  --save_path=svm\svm_1_24_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

<a id="32__svm"></a>
## 32       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_32_1  --save_path=svm\svm_1_32_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_32_2  --save_path=svm\svm_1_32_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1










