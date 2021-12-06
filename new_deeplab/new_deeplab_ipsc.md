<!-- MarkdownTOC -->

- [build_data](#build_dat_a_)
    - [g1       @ build_data](#g1___build_data_)
        - [patches       @ g1/build_data](#patches___g1_build_dat_a_)
        - [ipsc_multi       @ g1/build_data](#ipsc_multi___g1_build_dat_a_)
    - [g2_4       @ build_data](#g2_4___build_data_)
        - [ipsc_multi       @ g2_4/build_data](#ipsc_multi___g2_4_build_dat_a_)
        - [patches       @ g2_4/build_data](#patches___g2_4_build_dat_a_)
    - [g3_4       @ build_data](#g3_4___build_data_)
        - [ipsc_multi       @ g3_4/build_data](#ipsc_multi___g3_4_build_dat_a_)
        - [patches       @ g3_4/build_data](#patches___g3_4_build_dat_a_)
    - [g2       @ build_data](#g2___build_data_)
        - [ipsc_multi       @ g2/build_data](#ipsc_multi___g2_build_dat_a_)
        - [patches       @ g2/build_data](#patches___g2_build_dat_a_)
    - [g3       @ build_data](#g3___build_data_)
        - [ipsc_multi       @ g3/build_data](#ipsc_multi___g3_build_dat_a_)
        - [patches       @ g3/build_data](#patches___g3_build_dat_a_)
    - [g4       @ build_data](#g4___build_data_)
        - [ipsc_multi       @ g4/build_data](#ipsc_multi___g4_build_dat_a_)
        - [patches       @ g4/build_data](#patches___g4_build_dat_a_)
- [hnasnet](#hnasnet_)
    - [g1       @ hnasnet](#g1___hnasne_t_)
        - [ipsc_multi       @ g1/hnasnet](#ipsc_multi___g1_hnasnet_)
    - [g2_4       @ hnasnet](#g2_4___hnasne_t_)
        - [ipsc_multi       @ g2_4/hnasnet](#ipsc_multi___g2_4_hnasnet_)
        - [steps-100       @ g2_4/hnasnet](#steps_100___g2_4_hnasnet_)
        - [steps-200       @ g2_4/hnasnet](#steps_200___g2_4_hnasnet_)
        - [steps-500       @ g2_4/hnasnet](#steps_500___g2_4_hnasnet_)
        - [steps-1000       @ g2_4/hnasnet](#steps_1000___g2_4_hnasnet_)
        - [steps-2000       @ g2_4/hnasnet](#steps_2000___g2_4_hnasnet_)
        - [steps-5000       @ g2_4/hnasnet](#steps_5000___g2_4_hnasnet_)
        - [steps-20000       @ g2_4/hnasnet](#steps_20000___g2_4_hnasnet_)
        - [patches       @ g2_4/hnasnet](#patches___g2_4_hnasnet_)
    - [g3_4       @ hnasnet](#g3_4___hnasne_t_)
        - [ipsc_multi       @ g3_4/hnasnet](#ipsc_multi___g3_4_hnasnet_)
        - [patches       @ g3_4/hnasnet](#patches___g3_4_hnasnet_)
    - [g2       @ hnasnet](#g2___hnasne_t_)
        - [on_g3       @ g2/hnasnet](#on_g3___g2_hnasnet_)
        - [on_g4       @ g2/hnasnet](#on_g4___g2_hnasnet_)
        - [ipsc_multi       @ g2/hnasnet](#ipsc_multi___g2_hnasnet_)
        - [patches       @ g2/hnasnet](#patches___g2_hnasnet_)
    - [g3       @ hnasnet](#g3___hnasne_t_)
        - [on_g2       @ g3/hnasnet](#on_g2___g3_hnasnet_)
        - [on_g4       @ g3/hnasnet](#on_g4___g3_hnasnet_)
        - [ipsc_multi       @ g3/hnasnet](#ipsc_multi___g3_hnasnet_)
        - [patches       @ g3/hnasnet](#patches___g3_hnasnet_)
    - [g4       @ hnasnet](#g4___hnasne_t_)
        - [on_g2       @ g4/hnasnet](#on_g2___g4_hnasnet_)
        - [on_g3       @ g4/hnasnet](#on_g3___g4_hnasnet_)
        - [ipsc_multi       @ g4/hnasnet](#ipsc_multi___g4_hnasnet_)
        - [patches       @ g4/hnasnet](#patches___g4_hnasnet_)
- [resnet101](#resnet101_)
    - [g2_4       @ resnet101](#g2_4___resnet10_1_)
    - [ipsc_multi       @ resnet101](#ipsc_multi___resnet10_1_)
    - [patches       @ resnet101](#patches___resnet10_1_)

<!-- /MarkdownTOC -->

<a id="build_dat_a_"></a>
# build_data

<a id="g1___build_data_"></a>
## g1       @ build_data-->new_deeplab_ipsc

<a id="patches___g1_build_dat_a_"></a>
### patches       @ g1/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g1 preprocess=1 patches=1 root_dir=/data/ipsc_patches

<a id="ipsc_multi___g1_build_dat_a_"></a>
### ipsc_multi       @ g1/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g1 preprocess=0 root_dir=/data/ipsc_multi n_classes=3 multi=1

<a id="g2_4___build_data_"></a>
## g2_4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2_4 preprocess=0

<a id="ipsc_multi___g2_4_build_dat_a_"></a>
### ipsc_multi       @ g2_4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2_4 preprocess=0 root_dir=/data/ipsc_multi n_classes=3 multi=1

<a id="patches___g2_4_build_dat_a_"></a>
### patches       @ g2_4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2_4 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="g3_4___build_data_"></a>
## g3_4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3_4 preprocess=0

<a id="ipsc_multi___g3_4_build_dat_a_"></a>
### ipsc_multi       @ g3_4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3_4 preprocess=0 root_dir=/data/ipsc_multi n_classes=3 multi=1

<a id="patches___g3_4_build_dat_a_"></a>
### patches       @ g3_4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3_4 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="g2___build_data_"></a>
## g2       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2 preprocess=0

<a id="ipsc_multi___g2_build_dat_a_"></a>
### ipsc_multi       @ g2/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2 preprocess=0 root_dir=/data/ipsc_multi n_classes=3 multi=1

<a id="patches___g2_build_dat_a_"></a>
### patches       @ g2/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="g3___build_data_"></a>
## g3       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3 preprocess=0

<a id="ipsc_multi___g3_build_dat_a_"></a>
### ipsc_multi       @ g3/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3 preprocess=0 root_dir=/data/ipsc_multi n_classes=3 multi=1

<a id="patches___g3_build_dat_a_"></a>
### patches       @ g3/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="g4___build_data_"></a>
## g4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g4 preprocess=0

<a id="ipsc_multi___g4_build_dat_a_"></a>
### ipsc_multi       @ g4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g4 preprocess=0 root_dir=/data/ipsc_multi n_classes=3 multi=1


<a id="patches___g4_build_dat_a_"></a>
### patches       @ g4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g4 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="hnasnet_"></a>
# hnasnet

<a id="g1___hnasne_t_"></a>
## g1       @ hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g1,_train_:b2 start=2

<a id="ipsc_multi___g1_hnasnet_"></a>
### ipsc_multi       @ g1/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g1,_train_:b2 start=0

<a id="g2_4___hnasne_t_"></a>
## g2_4       @ hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g2_4,_train_:b2 start=2

<a id="ipsc_multi___g2_4_hnasnet_"></a>
### ipsc_multi       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2 start=1

<a id="steps_100___g2_4_hnasnet_"></a>
### steps-100       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2:steps-100 start=0

<a id="steps_200___g2_4_hnasnet_"></a>
### steps-200       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2:steps-200 start=0

<a id="steps_500___g2_4_hnasnet_"></a>
### steps-500       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2:steps-500 start=0

<a id="steps_1000___g2_4_hnasnet_"></a>
### steps-1000       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2:steps-1000 start=0

<a id="steps_2000___g2_4_hnasnet_"></a>
### steps-2000       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2:steps-2000 start=0

<a id="steps_5000___g2_4_hnasnet_"></a>
### steps-5000       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2:steps-5000 start=0

<a id="steps_20000___g2_4_hnasnet_"></a>
### steps-20000       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2:steps-20000 start=0

<a id="patches___g2_4_hnasnet_"></a>
### patches       @ g2_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g2_4,_train_:b2 start=1

<a id="g3_4___hnasne_t_"></a>
## g3_4       @ hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g3_4,_train_:b2 start=2

<a id="ipsc_multi___g3_4_hnasnet_"></a>
### ipsc_multi       @ g3_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g3_4,_train_:b2 start=0

<a id="patches___g3_4_hnasnet_"></a>
### patches       @ g3_4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g3_4,_train_:b2 start=1

<a id="g2___hnasne_t_"></a>
## g2       @ hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g2,_train_:b2 start=2

<a id="on_g3___g2_hnasnet_"></a>
### on_g3       @ g2/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g2:+++vis:g3,_train_:b2 start=1

<a id="on_g4___g2_hnasnet_"></a>
### on_g4       @ g2/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g2:+++vis:g4,_train_:b2 start=1

<a id="ipsc_multi___g2_hnasnet_"></a>
### ipsc_multi       @ g2/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2,_train_:b2 start=0

<a id="patches___g2_hnasnet_"></a>
### patches       @ g2/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g2,_train_:b2 start=1

<a id="g3___hnasne_t_"></a>
## g3       @ hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g3,_train_:b2 start=2

<a id="on_g2___g3_hnasnet_"></a>
### on_g2       @ g3/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g3:+++vis:g2,_train_:b2 start=1

<a id="on_g4___g3_hnasnet_"></a>
### on_g4       @ g3/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g3:+++vis:g4,_train_:b2 start=1

<a id="ipsc_multi___g3_hnasnet_"></a>
### ipsc_multi       @ g3/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g3,_train_:b2 start=0

<a id="patches___g3_hnasnet_"></a>
### patches       @ g3/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g3,_train_:b2 start=1

<a id="g4___hnasne_t_"></a>
## g4       @ hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g4,_train_:b2 start=2

<a id="on_g2___g4_hnasnet_"></a>
### on_g2       @ g4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g4:+++vis:g2,_train_:b2 start=1

<a id="on_g3___g4_hnasnet_"></a>
### on_g3       @ g4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g4:+++vis:g3,_train_:b2 start=1

<a id="ipsc_multi___g4_hnasnet_"></a>
### ipsc_multi       @ g4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g4,_train_:b2 start=0

<a id="patches___g4_hnasnet_"></a>
### patches       @ g4/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g4,_train_:b2 start=1

<a id="resnet101_"></a>
# resnet101

<a id="g2_4___resnet10_1_"></a>
## g2_4       @ resnet101-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_resnet_:atrous-6_12_18,_ipsc_:train:vis:g2_4,_train_:b2 start=0

<a id="ipsc_multi___resnet10_1_"></a>
## ipsc_multi       @ resnet101-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_resnet_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2 start=0

<a id="patches___resnet10_1_"></a>
## patches       @ resnet101-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_resnet_:atrous-6_12_18,_ipsc_:patches:train:vis:g2_4,_train_:b2 start=0
