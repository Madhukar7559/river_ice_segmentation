set -x 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_384_384_25_100_rot_15_345_4_flip/raw_segmentation_results --save_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_512_512_25_100_rot_15_345_4_flip/raw_segmentation_results --save_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1