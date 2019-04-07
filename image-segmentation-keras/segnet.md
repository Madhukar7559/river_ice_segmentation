# 512

CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_512_512_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" --epochs=1000 

## hml

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_512_512_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" --epochs=1000 

## evaluation 

### hml

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_512_512_25_100_rot_15_345_4_flip/weights_max_val_acc_352.h5 --test_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" 

## vis

#### hml

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1 --save_stitched=0

## validation

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_512_512_25_100_rot_15_345_4_flip/weights_max_val_acc_352.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/validation_0_20_512_512_512_512/images  --seg_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_512_512_512_512/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitching

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_512_512_512_512/predictions/raw stitched_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_512_512_512_512/predictions/stitched patch_height=512 patch_width=512 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" 

## videos

CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/YUN00001_0_239_512_512_512_512/images/" --output_path="/data/617/images/YUN00001_0_239_512_512_512_512/vgg_segnet_32_18_512_512_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" 

# 640

### hml

CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_49_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_49_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 

### evaluation 

#### hml

CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_305.h5 --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

### vis

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1 --save_stitched=0

### validation

#### hml

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_305.h5 --test_images="/data/617/images/validation_0_20_640_640_640_640/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/validation_0_20_640_640_640_640/images  --seg_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

### stitching

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/raw stitched_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

zr vgg_segnet_max_val_acc_validation_0_20_640_640_640_640_stitched /data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched

###  4

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --load_weights=1

### evaluation 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


### 8

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_7_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_7_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --load_weights=1

### evaluation 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


### 16

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --load_weights=1


### evaluation 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1



### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


### 24

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --load_weights=1

### evaluation 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


## 32

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000

### evaluation 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 CUDNN_PATH="/usr/local/cuda-8.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_189.h5 --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

### vis

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

### validation

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 CUDNN_PATH="/usr/local/cuda-8.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_189.h5 --test_images="/data/617/images/validation_0_20_640_640_640_640/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 


### stitching

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/raw stitched_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

zr vgg_segnet_max_val_acc_validation_0_20_640_640_640_640_stitched /data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched

# 640 - selective

## 4 - non_aug

### 2

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_2_rt2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=2 --load_weights=1

### 2 - 0_14

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_0_14_2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=2 --load_weights=1 --start_id=0 --end_id=14


### 10

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_10_rt2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=10 --load_weights=0

### 10 - 0_14

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_0_14_10 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=10 --load_weights=1 --start_id=0 --end_id=14


### 100

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_100_rt --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100 --load_weights=1

### 100 - 0_14

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_0_14_100 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100 --load_weights=1 --start_id=0 --end_id=14

### 1000

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_1000 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=1000 --load_weights=1

### 1000 - 0_14

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_0_14_1000 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=1000 --load_weights=1 --start_id=0 --end_id=14

## 4

### 2

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_2 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=2 --load_weights=1

### 5

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_5 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=5 --load_weights=1

### 10

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_10 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=10 --load_weights=1

### 100

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100 --load_weights=1

### vis

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

### 1K

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_1K --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=1000 --load_weights=1

### 5K

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_5K --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=5000 --load_weights=1

## 16

### 100

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100

## 32

### 100

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100
