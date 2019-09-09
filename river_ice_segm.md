<!-- MarkdownTOC -->

- [plotIceConcentration](#ploticeconcentratio_n_)
   - [training_32_49       @ plotIceConcentration](#training_32_49___ploticeconcentration_)
      - [deeplab       @ training_32_49/plotIceConcentration](#deeplab___training_32_49_ploticeconcentratio_n_)
         - [anchor       @ deeplab/training_32_49/plotIceConcentration](#anchor___deeplab_training_32_49_ploticeconcentratio_n_)
         - [frazil       @ deeplab/training_32_49/plotIceConcentration](#frazil___deeplab_training_32_49_ploticeconcentratio_n_)
      - [auto_deeplab       @ training_32_49/plotIceConcentration](#auto_deeplab___training_32_49_ploticeconcentratio_n_)
         - [anchor       @ auto_deeplab/training_32_49/plotIceConcentration](#anchor___auto_deeplab_training_32_49_ploticeconcentration_)
         - [frazil       @ auto_deeplab/training_32_49/plotIceConcentration](#frazil___auto_deeplab_training_32_49_ploticeconcentration_)
      - [resnet101_psp       @ training_32_49/plotIceConcentration](#resnet101_psp___training_32_49_ploticeconcentratio_n_)
         - [anchor       @ resnet101_psp/training_32_49/plotIceConcentration](#anchor___resnet101_psp_training_32_49_ploticeconcentratio_n_)
         - [frazil       @ resnet101_psp/training_32_49/plotIceConcentration](#frazil___resnet101_psp_training_32_49_ploticeconcentratio_n_)
      - [segnet       @ training_32_49/plotIceConcentration](#segnet___training_32_49_ploticeconcentratio_n_)
         - [max_acc       @ segnet/training_32_49/plotIceConcentration](#max_acc___segnet_training_32_49_ploticeconcentration_)
      - [unet       @ training_32_49/plotIceConcentration](#unet___training_32_49_ploticeconcentratio_n_)
      - [densenet       @ training_32_49/plotIceConcentration](#densenet___training_32_49_ploticeconcentratio_n_)
         - [anchor       @ densenet/training_32_49/plotIceConcentration](#anchor___densenet_training_32_49_ploticeconcentration_)
         - [frazil       @ densenet/training_32_49/plotIceConcentration](#frazil___densenet_training_32_49_ploticeconcentration_)
      - [svm       @ training_32_49/plotIceConcentration](#svm___training_32_49_ploticeconcentratio_n_)
         - [anchor       @ svm/training_32_49/plotIceConcentration](#anchor___svm_training_32_49_ploticeconcentratio_n_)
         - [frazil       @ svm/training_32_49/plotIceConcentration](#frazil___svm_training_32_49_ploticeconcentratio_n_)
      - [svm_deeplab       @ training_32_49/plotIceConcentration](#svm_deeplab___training_32_49_ploticeconcentratio_n_)
         - [no_labels       @ svm_deeplab/training_32_49/plotIceConcentration](#no_labels___svm_deeplab_training_32_49_ploticeconcentratio_n_)
      - [svm_deeplab_densenet       @ training_32_49/plotIceConcentration](#svm_deeplab_densenet___training_32_49_ploticeconcentratio_n_)
      - [svm_deeplab_unet_densenet_segnet       @ training_32_49/plotIceConcentration](#svm_deeplab_unet_densenet_segnet___training_32_49_ploticeconcentratio_n_)
         - [Combined       @ svm_deeplab_unet_densenet_segnet/training_32_49/plotIceConcentration](#combined___svm_deeplab_unet_densenet_segnet_training_32_49_ploticeconcentration_)
         - [anchor       @ svm_deeplab_unet_densenet_segnet/training_32_49/plotIceConcentration](#anchor___svm_deeplab_unet_densenet_segnet_training_32_49_ploticeconcentration_)
         - [frazil       @ svm_deeplab_unet_densenet_segnet/training_32_49/plotIceConcentration](#frazil___svm_deeplab_unet_densenet_segnet_training_32_49_ploticeconcentration_)
   - [training_4_49       @ plotIceConcentration](#training_4_49___ploticeconcentration_)
      - [svm_deeplab_unet_densenet_segnet       @ training_4_49/plotIceConcentration](#svm_deeplab_unet_densenet_segnet___training_4_49_ploticeconcentration_)
         - [Combined       @ svm_deeplab_unet_densenet_segnet/training_4_49/plotIceConcentration](#combined___svm_deeplab_unet_densenet_segnet_training_4_49_ploticeconcentratio_n_)
   - [video       @ plotIceConcentration](#video___ploticeconcentration_)
      - [YUN00001_3600       @ video/plotIceConcentration](#yun00001_3600___video_ploticeconcentration_)
         - [combined       @ YUN00001_3600/video/plotIceConcentration](#combined___yun00001_3600_video_ploticeconcentration_)
            - [svm       @ combined/YUN00001_3600/video/plotIceConcentration](#svm___combined_yun00001_3600_video_ploticeconcentratio_n_)
         - [frazil       @ YUN00001_3600/video/plotIceConcentration](#frazil___yun00001_3600_video_ploticeconcentration_)
            - [svm       @ frazil/YUN00001_3600/video/plotIceConcentration](#svm___frazil_yun00001_3600_video_ploticeconcentratio_n_)
         - [anchor       @ YUN00001_3600/video/plotIceConcentration](#anchor___yun00001_3600_video_ploticeconcentration_)
            - [svm       @ anchor/YUN00001_3600/video/plotIceConcentration](#svm___anchor_yun00001_3600_video_ploticeconcentratio_n_)
      - [20160122_YUN00002_700_2500       @ video/plotIceConcentration](#20160122_yun00002_700_2500___video_ploticeconcentration_)
         - [combined       @ 20160122_YUN00002_700_2500/video/plotIceConcentration](#combined___20160122_yun00002_700_2500_video_ploticeconcentratio_n_)
            - [plot_changed_seg_count       @ combined/20160122_YUN00002_700_2500/video/plotIceConcentration](#plot_changed_seg_count___combined_20160122_yun00002_700_2500_video_ploticeconcentration_)
         - [frazil       @ 20160122_YUN00002_700_2500/video/plotIceConcentration](#frazil___20160122_yun00002_700_2500_video_ploticeconcentratio_n_)
         - [anchor       @ 20160122_YUN00002_700_2500/video/plotIceConcentration](#anchor___20160122_yun00002_700_2500_video_ploticeconcentratio_n_)
      - [20160122_YUN00020_2000_3800       @ video/plotIceConcentration](#20160122_yun00020_2000_3800___video_ploticeconcentration_)
         - [combined       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration](#combined___20160122_yun00020_2000_3800_video_ploticeconcentration_)
            - [svm       @ combined/20160122_YUN00020_2000_3800/video/plotIceConcentration](#svm___combined_20160122_yun00020_2000_3800_video_ploticeconcentratio_n_)
         - [frazil       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration](#frazil___20160122_yun00020_2000_3800_video_ploticeconcentration_)
            - [svm       @ frazil/20160122_YUN00020_2000_3800/video/plotIceConcentration](#svm___frazil_20160122_yun00020_2000_3800_video_ploticeconcentratio_n_)
         - [anchor       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration](#anchor___20160122_yun00020_2000_3800_video_ploticeconcentration_)
            - [svm       @ anchor/20160122_YUN00020_2000_3800/video/plotIceConcentration](#svm___anchor_20160122_yun00020_2000_3800_video_ploticeconcentratio_n_)
      - [20161203_Deployment_1_YUN00001_900_2700       @ video/plotIceConcentration](#20161203_deployment_1_yun00001_900_2700___video_ploticeconcentration_)
         - [combined       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#combined___20161203_deployment_1_yun00001_900_2700_video_ploticeconcentration_)
            - [svm       @ combined/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#svm___combined_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_)
               - [20161203_Deployment_1_YUN00001_900_1200       @ svm/combined/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#20161203_deployment_1_yun00001_900_1200___svm_combined_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_)
         - [frazil       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#frazil___20161203_deployment_1_yun00001_900_2700_video_ploticeconcentration_)
            - [svm       @ frazil/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#svm___frazil_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_)
               - [20161203_Deployment_1_YUN00001_900_1200       @ svm/frazil/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#20161203_deployment_1_yun00001_900_1200___svm_frazil_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_)
         - [anchor       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#anchor___20161203_deployment_1_yun00001_900_2700_video_ploticeconcentration_)
            - [svm       @ anchor/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#svm___anchor_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_)
               - [20161203_Deployment_1_YUN00001_2000_2300       @ svm/anchor/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#20161203_deployment_1_yun00001_2000_2300___svm_anchor_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_)
               - [20161203_Deployment_1_YUN00001_900_1200       @ svm/anchor/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration](#20161203_deployment_1_yun00001_900_1200___svm_anchor_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_)
      - [20161203_Deployment_1_YUN00002_1800       @ video/plotIceConcentration](#20161203_deployment_1_yun00002_1800___video_ploticeconcentration_)
         - [combined       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration](#combined___20161203_deployment_1_yun00002_1800_video_ploticeconcentration_)
         - [frazil       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration](#frazil___20161203_deployment_1_yun00002_1800_video_ploticeconcentration_)
         - [anchor       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration](#anchor___20161203_deployment_1_yun00002_1800_video_ploticeconcentration_)
- [videoToImgSeq](#videotoimgseq_)
   - [1920x1080       @ videoToImgSeq](#1920x1080___videotoimgse_q_)
   - [4k       @ videoToImgSeq](#4k___videotoimgse_q_)
      - [YUN00001_3600       @ 4k/videoToImgSeq](#yun00001_3600___4k_videotoimgseq_)
      - [YUN00001_3600__win       @ 4k/videoToImgSeq](#yun00001_3600_win___4k_videotoimgseq_)
      - [20160121_YUN00002_2000       @ 4k/videoToImgSeq](#20160121_yun00002_2000___4k_videotoimgseq_)
      - [20161201_YUN00002_1800       @ 4k/videoToImgSeq](#20161201_yun00002_1800___4k_videotoimgseq_)
      - [20160122_YUN00002_700_2500       @ 4k/videoToImgSeq](#20160122_yun00002_700_2500___4k_videotoimgseq_)
      - [20160122_YUN00020_2000_3800       @ 4k/videoToImgSeq](#20160122_yun00020_2000_3800___4k_videotoimgseq_)
      - [20160122_YUN00020_2000_3800__win_pc       @ 4k/videoToImgSeq](#20160122_yun00020_2000_3800_win_pc___4k_videotoimgseq_)
      - [20161203_Deployment_1_YUN00001_900_2700       @ 4k/videoToImgSeq](#20161203_deployment_1_yun00001_900_2700___4k_videotoimgseq_)
      - [20161203_Deployment_1_YUN00001_900_1200_win_pc       @ 4k/videoToImgSeq](#20161203_deployment_1_yun00001_900_1200_win_pc___4k_videotoimgseq_)
      - [20161203_Deployment_1_YUN00001_2000_2300__win_pc       @ 4k/videoToImgSeq](#20161203_deployment_1_yun00001_2000_2300_win_pc___4k_videotoimgseq_)
      - [20161203_Deployment_1_YUN00002_1800       @ 4k/videoToImgSeq](#20161203_deployment_1_yun00002_1800___4k_videotoimgseq_)
      - [20170114_YUN00005_1800       @ 4k/videoToImgSeq](#20170114_yun00005_1800___4k_videotoimgseq_)
- [320x640](#320x640_)
- [256](#256_)
   - [all       @ 256](#all___25_6_)
      - [ppt       @ all/256](#ppt___all_25_6_)
      - [rotation_and_flipping       @ all/256](#rotation_and_flipping___all_25_6_)
      - [merging       @ all/256](#merging___all_25_6_)
   - [0-31       @ 256](#0_31___25_6_)
      - [merging       @ 0-31/256](#merging___0_31_256_)
   - [32-49       @ 256](#32_49___25_6_)
      - [merging       @ 32-49/256](#merging___32_49_25_6_)
   - [batch_all       @ 256](#batch_all___25_6_)
   - [validation       @ 256](#validation___25_6_)
      - [stitching       @ validation/256](#stitching___validation_256_)
   - [videos       @ 256](#videos___25_6_)
      - [stitching       @ videos/256](#stitching___videos_256_)
- [384](#384_)
   - [40/160       @ 384](#40_160___38_4_)
   - [25/100       @ 384](#25_100___38_4_)
   - [validation       @ 384](#validation___38_4_)
      - [stitching       @ validation/384](#stitching___validation_384_)
   - [videos       @ 384](#videos___38_4_)
      - [stitching       @ videos/384](#stitching___videos_384_)
   - [vis       @ 384](#vis___38_4_)
      - [unet       @ vis/384](#unet___vis_38_4_)
         - [hml       @ unet/vis/384](#hml___unet_vis_384_)
         - [weird       @ unet/vis/384](#weird___unet_vis_384_)
- [512](#512_)
   - [40/160       @ 512](#40_160___51_2_)
   - [25/100       @ 512](#25_100___51_2_)
   - [validation       @ 512](#validation___51_2_)
      - [stitching       @ validation/512](#stitching___validation_512_)
   - [videos       @ 512](#videos___51_2_)
      - [stitching       @ videos/512](#stitching___videos_512_)
- [640](#640_)
   - [64/256       @ 640](#64_256___64_0_)
   - [non_aug       @ 640](#non_aug___64_0_)
      - [0__3       @ non_aug/640](#0_3___non_aug_64_0_)
         - [sel-2       @ 0__3/non_aug/640](#sel_2___0_3_non_aug_64_0_)
         - [sel-10       @ 0__3/non_aug/640](#sel_10___0_3_non_aug_64_0_)
         - [sel-100       @ 0__3/non_aug/640](#sel_100___0_3_non_aug_64_0_)
         - [sel-1000       @ 0__3/non_aug/640](#sel_1000___0_3_non_aug_64_0_)
         - [sel-5000       @ 0__3/non_aug/640](#sel_5000___0_3_non_aug_64_0_)
      - [32_-49       @ non_aug/640](#32_49___non_aug_64_0_)
      - [0__49       @ non_aug/640](#0_49___non_aug_64_0_)
      - [4__49       @ non_aug/640](#4_49___non_aug_64_0_)
      - [entire_image       @ non_aug/640](#entire_image___non_aug_64_0_)
         - [0-3       @ entire_image/non_aug/640](#0_3___entire_image_non_aug_640_)
         - [0-7       @ entire_image/non_aug/640](#0_7___entire_image_non_aug_640_)
         - [0-15       @ entire_image/non_aug/640](#0_15___entire_image_non_aug_640_)
         - [0-23       @ entire_image/non_aug/640](#0_23___entire_image_non_aug_640_)
         - [0-31       @ entire_image/non_aug/640](#0_31___entire_image_non_aug_640_)
         - [32-49       @ entire_image/non_aug/640](#32_49___entire_image_non_aug_640_)
         - [4-49       @ entire_image/non_aug/640](#4_49___entire_image_non_aug_640_)
      - [ablation       @ non_aug/640](#ablation___non_aug_64_0_)
         - [0__3       @ ablation/non_aug/640](#0_3___ablation_non_aug_640_)
         - [sel-2       @ ablation/non_aug/640](#sel_2___ablation_non_aug_640_)
         - [sel-2       @ ablation/non_aug/640](#sel_2___ablation_non_aug_640__1)
         - [sel-10       @ ablation/non_aug/640](#sel_10___ablation_non_aug_640_)
         - [sel-100       @ ablation/non_aug/640](#sel_100___ablation_non_aug_640_)
         - [sel-1000       @ ablation/non_aug/640](#sel_1000___ablation_non_aug_640_)
         - [sel-5000       @ ablation/non_aug/640](#sel_5000___ablation_non_aug_640_)
   - [25/100       @ 640](#25_100___64_0_)
   - [validation       @ 640](#validation___64_0_)
      - [stitching       @ validation/640](#stitching___validation_640_)
   - [videos       @ 640](#videos___64_0_)
      - [stitching       @ videos/640](#stitching___videos_640_)
- [800](#800_)
   - [80/320       @ 800](#80_320___80_0_)
   - [non_aug       @ 800](#non_aug___80_0_)
      - [0__3       @ non_aug/800](#0_3___non_aug_80_0_)
      - [32__49       @ non_aug/800](#32_49___non_aug_80_0_)
      - [0__49       @ non_aug/800](#0_49___non_aug_80_0_)
      - [4__49       @ non_aug/800](#4_49___non_aug_80_0_)
      - [entire_image       @ non_aug/800](#entire_image___non_aug_80_0_)
         - [32-49       @ entire_image/non_aug/800](#32_49___entire_image_non_aug_800_)
         - [4-49       @ entire_image/non_aug/800](#4_49___entire_image_non_aug_800_)
      - [ablation       @ non_aug/800](#ablation___non_aug_80_0_)
   - [25/100       @ 800](#25_100___80_0_)
   - [video       @ 800](#video___80_0_)
- [1000](#100_0_)
   - [100/400       @ 1000](#100_400___1000_)
   - [video       @ 1000](#video___1000_)
      - [1920x1080       @ video/1000](#1920x1080___video_1000_)
- [stitch multiple results](#stitch_multiple_result_s_)
- [svm](#svm_)
   - [4       @ svm](#4___sv_m_)
   - [8       @ svm](#8___sv_m_)
   - [16       @ svm](#16___sv_m_)
   - [24       @ svm](#24___sv_m_)
   - [32       @ svm](#32___sv_m_)

<!-- /MarkdownTOC -->

<a id="ploticeconcentratio_n_"></a>
# plotIceConcentration

<a id="training_32_49___ploticeconcentration_"></a>
## training_32_49       @ plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training/images --labels_path=/data/617/images/training/labels --images_ext=tif --labels_ext=tif --n_classes=3

<a id="deeplab___training_32_49_ploticeconcentratio_n_"></a>
### deeplab       @ training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab

<a id="anchor___deeplab_training_32_49_ploticeconcentratio_n_"></a>
#### anchor       @ deeplab/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab --ice_type=1

<a id="frazil___deeplab_training_32_49_ploticeconcentratio_n_"></a>
#### frazil       @ deeplab/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab --ice_type=2

<a id="auto_deeplab___training_32_49_ploticeconcentratio_n_"></a>
### auto_deeplab       @ training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\nas_hnasnet_0_31_ade20k_training_32_49_raw__orca_190909_083554 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=auto_deeplab

<a id="anchor___auto_deeplab_training_32_49_ploticeconcentration_"></a>
#### anchor       @ auto_deeplab/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\nas_hnasnet_0_31_ade20k_training_32_49_raw__orca_190909_083554 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=auto_deeplab --ice_type=1

<a id="frazil___auto_deeplab_training_32_49_ploticeconcentration_"></a>
#### frazil       @ auto_deeplab/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\nas_hnasnet_0_31_ade20k_training_32_49_raw__orca_190909_083554 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=auto_deeplab --ice_type=2

<a id="resnet101_psp___training_32_49_ploticeconcentratio_n_"></a>
### resnet101_psp       @ training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\resnet_v1_101_0_31_training_32_49_raw_grs_190909_101731 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=resnet101_psp

<a id="anchor___resnet101_psp_training_32_49_ploticeconcentratio_n_"></a>
#### anchor       @ resnet101_psp/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\resnet_v1_101_0_31_training_32_49_raw_grs_190909_101731 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=resnet101_psp --ice_type=1

<a id="frazil___resnet101_psp_training_32_49_ploticeconcentratio_n_"></a>
#### frazil       @ resnet101_psp/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\resnet_v1_101_0_31_training_32_49_raw_grs_190909_101731 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=resnet101_psp --ice_type=2

<a id="segnet___training_32_49_ploticeconcentratio_n_"></a>
### segnet       @ training_32_49/plotIceConcentration

<a id="max_acc___segnet_training_32_49_ploticeconcentration_"></a>
#### max_acc       @ segnet/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log\segnet --seg_paths=log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_acc_raw_grs_190524_145647,log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_raw_grs_190524_154518,log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_min_loss_raw_grs_190524_154535 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=max_acc,max_val_acc,min_loss


<a id="unet___training_32_49_ploticeconcentratio_n_"></a>
### unet       @ training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=unet\log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_grs_190413_102443\raw --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=unet

<a id="densenet___training_32_49_ploticeconcentratio_n_"></a>
### densenet       @ training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet

<a id="anchor___densenet_training_32_49_ploticeconcentration_"></a>
#### anchor       @ densenet/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet --ice_type=1

<a id="frazil___densenet_training_32_49_ploticeconcentration_"></a>
#### frazil       @ densenet/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet --ice_type=2


<a id="svm___training_32_49_ploticeconcentratio_n_"></a>
### svm       @ training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_1 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_1

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2


<a id="anchor___svm_training_32_49_ploticeconcentratio_n_"></a>
#### anchor       @ svm/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2 --ice_type=1

<a id="frazil___svm_training_32_49_ploticeconcentratio_n_"></a>
#### frazil       @ svm/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2 --ice_type=2

<a id="svm_deeplab___training_32_49_ploticeconcentratio_n_"></a>
### svm_deeplab       @ training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab --ice_type=1

<a id="no_labels___svm_deeplab_training_32_49_ploticeconcentratio_n_"></a>
#### no_labels       @ svm_deeplab/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab --ice_type=1

<a id="svm_deeplab_densenet___training_32_49_ploticeconcentratio_n_"></a>
### svm_deeplab_densenet       @ training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424,densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,densenet --ice_type=0

<a id="svm_deeplab_unet_densenet_segnet___training_32_49_ploticeconcentratio_n_"></a>
### svm_deeplab_unet_densenet_segnet       @ training_32_49/plotIceConcentration

<a id="combined___svm_deeplab_unet_densenet_segnet_training_32_49_ploticeconcentration_"></a>
#### Combined       @ svm_deeplab_unet_densenet_segnet/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424,unet\log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_grs_190413_102443\raw,densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638,segnet/log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_acc_raw_grs_190524_145647 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=0

<a id="anchor___svm_deeplab_unet_densenet_segnet_training_32_49_ploticeconcentration_"></a>
#### anchor       @ svm_deeplab_unet_densenet_segnet/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424,unet\log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_grs_190413_102443\raw,densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638,segnet/log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_acc_raw_grs_190524_145647 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=1

<a id="frazil___svm_deeplab_unet_densenet_segnet_training_32_49_ploticeconcentration_"></a>
#### frazil       @ svm_deeplab_unet_densenet_segnet/training_32_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424,unet\log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_grs_190413_102443\raw,densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638,segnet/log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_acc_raw_grs_190524_145647 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=2

<a id="training_4_49___ploticeconcentration_"></a>
## training_4_49       @ plotIceConcentration

<a id="svm_deeplab_unet_densenet_segnet___training_4_49_ploticeconcentration_"></a>
### svm_deeplab_unet_densenet_segnet       @ training_4_49/plotIceConcentration

<a id="combined___svm_deeplab_unet_densenet_segnet_training_4_49_ploticeconcentratio_n_"></a>
#### Combined       @ svm_deeplab_unet_densenet_segnet/training_4_49/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/training_4_49/images --labels_path=/data/617/images/training_4_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_4_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_3_training_4_49_raw_grs_190524_173504,unet\log_vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_val_acc_raw_grs_190524_172038,densenet\log_rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu_predict_acc_training_4_49_raw_z370_190524_173744,segnet/log_vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_acc_raw_grs_190524_173006 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=0

python3 plotIceConcentration.py --images_path=/data/617/images/training_4_49/images --labels_path=/data/617/images/training_4_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_4_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_3_training_4_49_raw_grs_190524_173504,unet\log_vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_val_acc_raw_grs_190524_172038,densenet\log_rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu_predict_acc_training_4_49_raw_z370_190524_173744,segnet/log_vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_acc_raw_grs_190524_173006 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=1

python3 plotIceConcentration.py --images_path=/data/617/images/training_4_49/images --labels_path=/data/617/images/training_4_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_4_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_3_training_4_49_raw_grs_190524_173504,unet\log_vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_val_acc_raw_grs_190524_172038,densenet\log_rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu_predict_acc_training_4_49_raw_z370_190524_173744,segnet/log_vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_acc_raw_grs_190524_173006 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=2



<a id="video___ploticeconcentration_"></a>
## video       @ plotIceConcentration


<a id="yun00001_3600___video_ploticeconcentration_"></a>
### YUN00001_3600       @ video/plotIceConcentration

<a id="combined___yun00001_3600_video_ploticeconcentration_"></a>
#### combined       @ YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=deeplab/log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,unet/log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,densenet/log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/YUN00001_3600  --out_size=1920x720 --enable_plotting=0

<a id="svm___combined_yun00001_3600_video_ploticeconcentratio_n_"></a>
##### svm       @ combined/YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/YUN00001_3600_deeplab  --out_size=1920x720 --end_id=899

<a id="frazil___yun00001_3600_video_ploticeconcentration_"></a>
#### frazil       @ YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/YUN00001_3600  --out_size=1920x720 

<a id="svm___frazil_yun00001_3600_video_ploticeconcentratio_n_"></a>
##### svm       @ frazil/YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/YUN00001_3600_deeplab  --out_size=1920x720 --end_id=899


<a id="anchor___yun00001_3600_video_ploticeconcentration_"></a>
#### anchor       @ YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/YUN00001_3600  --out_size=1920x720 

<a id="svm___anchor_yun00001_3600_video_ploticeconcentratio_n_"></a>
##### svm       @ anchor/YUN00001_3600/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/YUN00001_3600_deeplab  --out_size=1920x720 --end_id=899


<a id="20160122_yun00002_700_2500___video_ploticeconcentration_"></a>
### 20160122_YUN00002_700_2500       @ video/plotIceConcentration

<a id="combined___20160122_yun00002_700_2500_video_ploticeconcentratio_n_"></a>
#### combined       @ 20160122_YUN00002_700_2500/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="plot_changed_seg_count___combined_20160122_yun00002_700_2500_video_ploticeconcentration_"></a>
##### plot_changed_seg_count       @ combined/20160122_YUN00002_700_2500/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 --plot_changed_seg_count=1

<a id="frazil___20160122_yun00002_700_2500_video_ploticeconcentratio_n_"></a>
#### frazil       @ 20160122_YUN00002_700_2500/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="anchor___20160122_yun00002_700_2500_video_ploticeconcentratio_n_"></a>
#### anchor       @ 20160122_YUN00002_700_2500/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="20160122_yun00020_2000_3800___video_ploticeconcentration_"></a>
### 20160122_YUN00020_2000_3800       @ video/plotIceConcentration

<a id="combined___20160122_yun00020_2000_3800_video_ploticeconcentration_"></a>
#### combined       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 

<a id="svm___combined_20160122_yun00020_2000_3800_video_ploticeconcentratio_n_"></a>
##### svm       @ combined/20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00020_2000_3800_deeplab  --out_size=1920x720 --end_id=299


<a id="frazil___20160122_yun00020_2000_3800_video_ploticeconcentration_"></a>
#### frazil       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720

<a id="svm___frazil_20160122_yun00020_2000_3800_video_ploticeconcentratio_n_"></a>
##### svm       @ frazil/20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00020_2000_3800_deeplab  --out_size=1920x720 --end_id=299


<a id="anchor___20160122_yun00020_2000_3800_video_ploticeconcentration_"></a>
#### anchor       @ 20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 

<a id="svm___anchor_20160122_yun00020_2000_3800_video_ploticeconcentratio_n_"></a>
##### svm       @ anchor/20160122_YUN00020_2000_3800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00020_2000_3800_deeplab  --out_size=1920x720 --end_id=299



<a id="20161203_deployment_1_yun00001_900_2700___video_ploticeconcentration_"></a>
### 20161203_Deployment_1_YUN00001_900_2700       @ video/plotIceConcentration

<a id="combined___20161203_deployment_1_yun00001_900_2700_video_ploticeconcentration_"></a>
#### combined       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm___combined_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_"></a>
##### svm       @ combined/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299

<a id="20161203_deployment_1_yun00001_900_1200___svm_combined_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_"></a>
###### 20161203_Deployment_1_YUN00001_900_1200       @ svm/combined/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_900_1200_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_900_1200_deeplab  --out_size=1920x720 --start_id=0 --end_id=299

<a id="frazil___20161203_deployment_1_yun00001_900_2700_video_ploticeconcentration_"></a>
#### frazil       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm___frazil_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_"></a>
##### svm       @ frazil/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_900_1200_deeplab  --out_size=1920x720 --start_id=0 --end_id=299

<a id="20161203_deployment_1_yun00001_900_1200___svm_frazil_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_"></a>
###### 20161203_Deployment_1_YUN00001_900_1200       @ svm/frazil/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_900_1200_deeplab  --out_size=1920x720 --start_id=0 --end_id=299


<a id="anchor___20161203_deployment_1_yun00001_900_2700_video_ploticeconcentration_"></a>
#### anchor       @ 20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm___anchor_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_"></a>
##### svm       @ anchor/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

<a id="20161203_deployment_1_yun00001_2000_2300___svm_anchor_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_"></a>
###### 20161203_Deployment_1_YUN00001_2000_2300       @ svm/anchor/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299

<a id="20161203_deployment_1_yun00001_900_1200___svm_anchor_20161203_deployment_1_yun00001_900_2700_video_ploticeconcentratio_n_"></a>
###### 20161203_Deployment_1_YUN00001_900_1200       @ svm/anchor/20161203_Deployment_1_YUN00001_900_2700/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299



<a id="20161203_deployment_1_yun00002_1800___video_ploticeconcentration_"></a>
### 20161203_Deployment_1_YUN00002_1800       @ video/plotIceConcentration

<a id="combined___20161203_deployment_1_yun00002_1800_video_ploticeconcentration_"></a>
#### combined       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 

<a id="frazil___20161203_deployment_1_yun00002_1800_video_ploticeconcentration_"></a>
#### frazil       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 

<a id="anchor___20161203_deployment_1_yun00002_1800_video_ploticeconcentration_"></a>
#### anchor       @ 20161203_Deployment_1_YUN00002_1800/video/plotIceConcentration

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 

<a id="videotoimgseq_"></a>
# videoToImgSeq

<a id="1920x1080___videotoimgse_q_"></a>
## 1920x1080       @ videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=0.50 dst_dir=/data/617/images/YUN00001_1920x1080/images

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_fa ctor=0.50 dst_dir=/data/617/images/YUN00002_1920x1080/images

<a id="4k___videotoimgse_q_"></a>
## 4k       @ videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=-1 resize_factor=1 dst_dir=/data/617/images/YUN00001/images

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=1 dst_dir=/data/617/images/YUN00001_1800/images

<a id="yun00001_3600___4k_videotoimgseq_"></a>
### YUN00001_3600       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=3600 resize_factor=1 dst_dir=/data/617/images/YUN00001_3600/images

<a id="yun00001_3600_win___4k_videotoimgseq_"></a>
### YUN00001_3600__win       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=3600 resize_factor=1 dst_dir=E:\Datasets\617\images\YUN00001_3600\images

<a id="20160121_yun00002_2000___4k_videotoimgseq_"></a>
### 20160121_YUN00002_2000       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=2000 dst_dir=/data/617/images/20160121_YUN00002_2000/images 

<a id="20161201_yun00002_1800___4k_videotoimgseq_"></a>
### 20161201_YUN00002_1800       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161201 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161201_YUN00002_1800/images

<a id="20160122_yun00002_700_2500___4k_videotoimgseq_"></a>
### 20160122_YUN00002_700_2500       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160122 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=700 dst_dir=/data/617/images/20160122_YUN00002_700_2500/images 

<a id="20160122_yun00020_2000_3800___4k_videotoimgseq_"></a>
### 20160122_YUN00020_2000_3800       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160122 seq_name=YUN00020 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=2000 dst_dir=/data/617/images/20160122_YUN00020_2000_3800/images

<a id="20160122_yun00020_2000_3800_win_pc___4k_videotoimgseq_"></a>
### 20160122_YUN00020_2000_3800__win_pc       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20160122 seq_name=YUN00020 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=2000 dst_dir=P:\Datasets\617\images\20160122_YUN00020_2000_300\images

<a id="20161203_deployment_1_yun00001_900_2700___4k_videotoimgseq_"></a>
### 20161203_Deployment_1_YUN00001_900_2700       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=900 dst_dir=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images

<a id="20161203_deployment_1_yun00001_900_1200_win_pc___4k_videotoimgseq_"></a>
### 20161203_Deployment_1_YUN00001_900_1200_win_pc       @ 4k/videoToImgSeq

__corrected__
python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=900 dst_dir=P:\Datasets\617\images\20161203_Deployment_1_YUN00001_900_1200\images

<a id="20161203_deployment_1_yun00001_2000_2300_win_pc___4k_videotoimgseq_"></a>
### 20161203_Deployment_1_YUN00001_2000_2300__win_pc       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=2000 dst_dir=E:\Datasets\617\images\20161203_Deployment_1_YUN00001_2000_2300\images

<a id="20161203_deployment_1_yun00002_1800___4k_videotoimgseq_"></a>
### 20161203_Deployment_1_YUN00002_1800       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161203 seq_name=Deployment_1_YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161203_Deployment_1_YUN00002_1800/images


<a id="20170114_yun00005_1800___4k_videotoimgseq_"></a>
### 20170114_YUN00005_1800       @ 4k/videoToImgSeq

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20170114 seq_name=YUN00005 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20170114_YUN00005_1800/images 

<a id="320x640_"></a>
# 320x640

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=320 patch_width=640 min_stride=25 max_stride=100

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=320 patch_width=640 min_stride=100 max_stride=200

<a id="256_"></a>
# 256


<a id="all___25_6_"></a>
## all       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=100 max_stride=200

<a id="ppt___all_25_6_"></a>
### ppt       @ all/256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1 start_id=2 end_id=2

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1 start_id=2 end_id=2 enable_rot=1 min_rot=15 max_rot=90


<a id="rotation_and_flipping___all_25_6_"></a>
### rotation_and_flipping       @ all/256

python3 subPatchDataset.py db_root_dir=/home/abhineet/N/Datasets/617 seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=20 max_rot=75

python3 subPatchDataset.py db_root_dir=/home/abhineet/N/Datasets/617 seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=20 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=100 max_stride=200 enable_rot=1 min_rot=90 max_rot=180 enable_flip=1

<a id="merging___all_25_6_"></a>
### merging       @ all/256

python3 mergeDatasets.py training_256_256_100_200_flip training_256_256_100_200_rot_90_180_flip 

<a id="0_31___25_6_"></a>
## 0-31       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=0 end_id=31

<a id="merging___0_31_256_"></a>
### merging       @ 0-31/256
python3 mergeDatasets.py training_0_31_256_256_25_100_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_126_235_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_15_125_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_236_345_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

<a id="32_49___25_6_"></a>
## 32-49       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=32 end_id=49

<a id="merging___32_49_25_6_"></a>
### merging       @ 32-49/256

python3 mergeDatasets.py training_32_49_256_256_25_100_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_126_235_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_15_125_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_236_345_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

<a id="batch_all___25_6_"></a>
## batch_all       @ 256

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

<a id="validation___25_6_"></a>
## validation       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=1 end_id=1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

<a id="stitching___validation_256_"></a>
### stitching       @ validation/256

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_1_1_256_256_256_256 patch_height=256 patch_width=256 start_id=1 end_id=1 patch_seq_type=images show_img=1 stacked=1 method=1 resize_factor=0.5

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_1_1_256_256_256_256 patch_height=256 patch_width=256 start_id=1 end_id=1 patch_seq_type=labels_deeplab_xception show_img=1 stacked=1 method=1 resize_factor=0.5

<a id="videos___25_6_"></a>
## videos       @ 256

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching___videos_256_"></a>
### stitching       @ videos/256

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1 

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1


<a id="384_"></a>
# 384

<a id="40_160___38_4_"></a>
## 40/160       @ 384

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49



<a id="25_100___38_4_"></a>
## 25/100       @ 384

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

<a id="validation___38_4_"></a>
## validation       @ 384

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=20


<a id="stitching___validation_384_"></a>
### stitching       @ validation/384

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_384_384_384_384 patch_height=384 patch_width=384 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos___38_4_"></a>
## videos       @ 384

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching___videos_384_"></a>
### stitching       @ videos/384

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_384_384_384_384 patch_height=384 patch_width=384 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1

<a id="vis___38_4_"></a>
## vis       @ 384

<a id="unet___vis_38_4_"></a>
### unet       @ vis/384

<a id="hml___unet_vis_384_"></a>
#### hml       @ unet/vis/384

python3 visDataset.py --images_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/images --labels_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/vis --n_classes=3 --start_id=0 --end_id=-1

python3 visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="weird___unet_vis_384_"></a>
#### weird       @ unet/vis/384

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1


python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_50_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=50 patch_seq_type=images show_img=1 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=vgg_unet2_max_val_acc_validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0 method=1
python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=fcn32_max_mean_acc_validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0 method=1


<a id="512_"></a>
# 512

<a id="40_160___51_2_"></a>
## 40/160       @ 512


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="25_100___51_2_"></a>
## 25/100       @ 512


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="validation___51_2_"></a>
## validation       @ 512

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=20

<a id="stitching___validation_512_"></a>
### stitching       @ validation/512

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_512_512_512_512 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos___51_2_"></a>
## videos       @ 512

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching___videos_512_"></a>
### stitching       @ videos/512

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_512_512_512_512 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1

<a id="640_"></a>
# 640

<a id="64_256___64_0_"></a>
## 64/256       @ 640

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="non_aug___64_0_"></a>
## non_aug       @ 640

<a id="0_3___non_aug_64_0_"></a>
### 0__3       @ non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

<a id="sel_2___0_3_non_aug_64_0_"></a>
#### sel-2       @ 0__3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=2

<a id="sel_10___0_3_non_aug_64_0_"></a>
#### sel-10       @ 0__3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=10

<a id="sel_100___0_3_non_aug_64_0_"></a>
#### sel-100       @ 0__3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=100

<a id="sel_1000___0_3_non_aug_64_0_"></a>
#### sel-1000       @ 0__3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=1000

<a id="sel_5000___0_3_non_aug_64_0_"></a>
#### sel-5000       @ 0__3/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=5000

<a id="32_49___non_aug_64_0_"></a>
### 32_-49       @ non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="0_49___non_aug_64_0_"></a>
### 0__49       @ non_aug/640

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=49 img_ext=tif

<a id="4_49___non_aug_64_0_"></a>
### 4__49       @ non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="entire_image___non_aug_64_0_"></a>
### entire_image       @ non_aug/640

<a id="0_3___entire_image_non_aug_640_"></a>
#### 0-3       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

<a id="0_7___entire_image_non_aug_640_"></a>
#### 0-7       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=7 img_ext=tif

<a id="0_15___entire_image_non_aug_640_"></a>
#### 0-15       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=15 img_ext=tif

<a id="0_23___entire_image_non_aug_640_"></a>
#### 0-23       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=23 img_ext=tif

<a id="0_31___entire_image_non_aug_640_"></a>
#### 0-31       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=31 img_ext=tif

<a id="32_49___entire_image_non_aug_640_"></a>
#### 32-49       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="4_49___entire_image_non_aug_640_"></a>
#### 4-49       @ entire_image/non_aug/640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="ablation___non_aug_64_0_"></a>
### ablation       @ non_aug/640

<a id="0_3___ablation_non_aug_640_"></a>
#### 0__3       @ ablation/non_aug/640

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=3

<a id="sel_2___ablation_non_aug_640_"></a>
#### sel-2       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=2

<a id="sel_2___ablation_non_aug_640__1"></a>
#### sel-2       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=2

<a id="sel_10___ablation_non_aug_640_"></a>
#### sel-10       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=10

<a id="sel_100___ablation_non_aug_640_"></a>
#### sel-100       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=100

<a id="sel_1000___ablation_non_aug_640_"></a>
#### sel-1000       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=1000

<a id="sel_5000___ablation_non_aug_640_"></a>
#### sel-5000       @ ablation/non_aug/640

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=5000

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=7

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=15

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=23

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

<a id="25_100___64_0_"></a>
## 25/100       @ 640

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49


python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

<a id="validation___64_0_"></a>
## validation       @ 640

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=20

<a id="stitching___validation_640_"></a>
### stitching       @ validation/640

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos___64_0_"></a>
## videos       @ 640

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_3600 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700_2500 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00020_2000_3800 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00001_900_2700 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00002_1800 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


<a id="stitching___videos_640_"></a>
### stitching       @ videos/640

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1


<a id="800_"></a>
# 800

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

<a id="80_320___80_0_"></a>
## 80/320       @ 800

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="non_aug___80_0_"></a>
## non_aug       @ 800

<a id="0_3___non_aug_80_0_"></a>
### 0__3       @ non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

<a id="32_49___non_aug_80_0_"></a>
### 32__49       @ non_aug/800

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="0_49___non_aug_80_0_"></a>
### 0__49       @ non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=49 img_ext=tif

<a id="4_49___non_aug_80_0_"></a>
### 4__49       @ non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="entire_image___non_aug_80_0_"></a>
### entire_image       @ non_aug/800

<a id="32_49___entire_image_non_aug_800_"></a>
#### 32-49       @ entire_image/non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="4_49___entire_image_non_aug_800_"></a>
#### 4-49       @ entire_image/non_aug/800

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="ablation___non_aug_80_0_"></a>
### ablation       @ non_aug/800

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=3

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=7

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=15

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=23

<a id="25_100___80_0_"></a>
## 25/100       @ 800

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="video___80_0_"></a>
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

<a id="100_0_"></a>
# 1000

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0


<a id="100_400___1000_"></a>
## 100/400       @ 1000

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="video___1000_"></a>
## video       @ 1000

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_2000 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20161201_YUN00002 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20170114_YUN00005 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="1920x1080___video_1000_"></a>
### 1920x1080       @ video/1000

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_1920x1080 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_1920x1080 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


<a id="stitch_multiple_result_s_"></a>
# stitch multiple results

python3 stitchMultipleResults.py --seg_root_dir=/home/abhineet/H/UofA/617/Project/presentation --images_path=/data/617/images/validation/images --save_path=/home/abhineet/H/UofA/617/Project/presentation/stitched --show_img=1 --resize_factor=0.25

<a id="svm_"></a>
# svm

<a id="4___sv_m_"></a>
## 4       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_4_1  --save_path=svm\svm_1_4_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_4_2  --save_path=svm\svm_1_4_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

<a id="8___sv_m_"></a>
## 8       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_8_1  --save_path=svm\svm_1_8_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_8_2  --save_path=svm\svm_1_8_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1


<a id="16___sv_m_"></a>
## 16       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_16_1  --save_path=svm\svm_1_16_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_16_2  --save_path=svm\svm_1_16_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

<a id="24___sv_m_"></a>
## 24       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_24_1  --save_path=svm\svm_1_24_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_24_2  --save_path=svm\svm_1_24_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

<a id="32___sv_m_"></a>
## 32       @ svm

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_32_1  --save_path=svm\svm_1_32_1\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1

python3 visDataset.py --images_path=E:\Datasets\617\images\training_32_49\images --labels_path=E:\Datasets\617\images\training_32_49\labels --seg_path=svm\svm_1_32_2  --save_path=svm\svm_1_32_2\vis --n_classes=3 --start_id=0 --end_id=-1 --stitch=1 --save_stitched=1










