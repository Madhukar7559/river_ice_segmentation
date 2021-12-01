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
        - [patches       @ g2/build_data](#patches___g2_build_dat_a_)
    - [g3       @ build_data](#g3___build_data_)
        - [patches       @ g3/build_data](#patches___g3_build_dat_a_)
    - [g4       @ build_data](#g4___build_data_)
        - [patches       @ g4/build_data](#patches___g4_build_dat_a_)
- [hnasnet](#hnasnet_)
    - [atrous:6_12_18       @ hnasnet](#atrous_6_12_18___hnasne_t_)
        - [g2_4       @ atrous:6_12_18/hnasnet](#g2_4___atrous_6_12_18_hnasnet_)
        - [ipsc_multi       @ atrous:6_12_18/hnasnet](#ipsc_multi___atrous_6_12_18_hnasnet_)
        - [patches       @ atrous:6_12_18/hnasnet](#patches___atrous_6_12_18_hnasnet_)
        - [g3_4       @ atrous:6_12_18/hnasnet](#g3_4___atrous_6_12_18_hnasnet_)
        - [patches       @ atrous:6_12_18/hnasnet](#patches___atrous_6_12_18_hnasnet__1)
        - [g2       @ atrous:6_12_18/hnasnet](#g2___atrous_6_12_18_hnasnet_)
        - [on_g3       @ atrous:6_12_18/hnasnet](#on_g3___atrous_6_12_18_hnasnet_)
        - [on_g4       @ atrous:6_12_18/hnasnet](#on_g4___atrous_6_12_18_hnasnet_)
        - [patches       @ atrous:6_12_18/hnasnet](#patches___atrous_6_12_18_hnasnet__2)
        - [g3       @ atrous:6_12_18/hnasnet](#g3___atrous_6_12_18_hnasnet_)
        - [on_g2       @ atrous:6_12_18/hnasnet](#on_g2___atrous_6_12_18_hnasnet_)
        - [on_g4       @ atrous:6_12_18/hnasnet](#on_g4___atrous_6_12_18_hnasnet__1)
        - [patches       @ atrous:6_12_18/hnasnet](#patches___atrous_6_12_18_hnasnet__3)
        - [g4       @ atrous:6_12_18/hnasnet](#g4___atrous_6_12_18_hnasnet_)
        - [on_g2       @ atrous:6_12_18/hnasnet](#on_g2___atrous_6_12_18_hnasnet__1)
        - [on_g3       @ atrous:6_12_18/hnasnet](#on_g3___atrous_6_12_18_hnasnet__1)
        - [patches       @ atrous:6_12_18/hnasnet](#patches___atrous_6_12_18_hnasnet__4)

<!-- /MarkdownTOC -->

<a id="build_dat_a_"></a>
# build_data

<a id="g1___build_data_"></a>
## g1       @ build_data-->new_deeplab_ipsc

<a id="patches___g1_build_dat_a_"></a>
### patches       @ g1/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g1 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="ipsc_multi___g1_build_dat_a_"></a>
### ipsc_multi       @ g1/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g1 preprocess=1  root_dir=/data/ipsc_multi n_classes=3 

<a id="g2_4___build_data_"></a>
## g2_4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2_4 preprocess=0

<a id="ipsc_multi___g2_4_build_dat_a_"></a>
### ipsc_multi       @ g2_4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2_4 preprocess=1 root_dir=/data/ipsc_multi n_classes=3 

<a id="patches___g2_4_build_dat_a_"></a>
### patches       @ g2_4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2_4 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="g3_4___build_data_"></a>
## g3_4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3_4 preprocess=0

<a id="ipsc_multi___g3_4_build_dat_a_"></a>
### ipsc_multi       @ g3_4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3_4 preprocess=1 root_dir=/data/ipsc_multi n_classes=3 

<a id="patches___g3_4_build_dat_a_"></a>
### patches       @ g3_4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3_4 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="g2___build_data_"></a>
## g2       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2 preprocess=0

<a id="patches___g2_build_dat_a_"></a>
### patches       @ g2/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="g3___build_data_"></a>
## g3       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3 preprocess=0
<a id="patches___g3_build_dat_a_"></a>
### patches       @ g3/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="g4___build_data_"></a>
## g4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g4 preprocess=0

<a id="patches___g4_build_dat_a_"></a>
### patches       @ g4/build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g4 preprocess=0 patches=1 root_dir=/data/ipsc_patches

<a id="hnasnet_"></a>
# hnasnet

<a id="atrous_6_12_18___hnasne_t_"></a>
## atrous:6_12_18       @ hnasnet-->new_deeplab_ipsc

<a id="g2_4___atrous_6_12_18_hnasnet_"></a>
### g2_4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g2_4,_train_:b2 start=2

<a id="ipsc_multi___atrous_6_12_18_hnasnet_"></a>
### ipsc_multi       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:multi:train:vis:g2_4,_train_:b2 start=1

<a id="patches___atrous_6_12_18_hnasnet_"></a>
### patches       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g2_4,_train_:b2 start=1

<a id="g3_4___atrous_6_12_18_hnasnet_"></a>
### g3_4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g3_4,_train_:b2 start=2

<a id="patches___atrous_6_12_18_hnasnet__1"></a>
### patches       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g3_4,_train_:b2 start=1

<a id="g2___atrous_6_12_18_hnasnet_"></a>
### g2       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g2,_train_:b2 start=2

<a id="on_g3___atrous_6_12_18_hnasnet_"></a>
### on_g3       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g2:+++vis:g3,_train_:b2 start=1

<a id="on_g4___atrous_6_12_18_hnasnet_"></a>
### on_g4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g2:+++vis:g4,_train_:b2 start=1

<a id="patches___atrous_6_12_18_hnasnet__2"></a>
### patches       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g2,_train_:b2 start=1

<a id="g3___atrous_6_12_18_hnasnet_"></a>
### g3       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g3,_train_:b2 start=2

<a id="on_g2___atrous_6_12_18_hnasnet_"></a>
### on_g2       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g3:+++vis:g2,_train_:b2 start=1

<a id="on_g4___atrous_6_12_18_hnasnet__1"></a>
### on_g4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g3:+++vis:g4,_train_:b2 start=1

<a id="patches___atrous_6_12_18_hnasnet__3"></a>
### patches       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g3,_train_:b2 start=1

<a id="g4___atrous_6_12_18_hnasnet_"></a>
### g4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g4,_train_:b2 start=2

<a id="on_g2___atrous_6_12_18_hnasnet__1"></a>
### on_g2       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g4:+++vis:g2,_train_:b2 start=1

<a id="on_g3___atrous_6_12_18_hnasnet__1"></a>
### on_g3       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g4:+++vis:g3,_train_:b2 start=1

<a id="patches___atrous_6_12_18_hnasnet__4"></a>
### patches       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:patches:train:vis:g4,_train_:b2 start=1

