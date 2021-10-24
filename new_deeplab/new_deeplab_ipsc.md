<!-- MarkdownTOC -->

- [build_data](#build_dat_a_)
    - [g2_4       @ build_data](#g2_4___build_data_)
    - [g3_4       @ build_data](#g3_4___build_data_)
    - [g2       @ build_data](#g2___build_data_)
    - [g3       @ build_data](#g3___build_data_)
    - [g4       @ build_data](#g4___build_data_)
- [hnasnet](#hnasnet_)
    - [atrous:6_12_18       @ hnasnet](#atrous_6_12_18___hnasne_t_)
        - [g2_4       @ atrous:6_12_18/hnasnet](#g2_4___atrous_6_12_18_hnasnet_)
        - [g3_4       @ atrous:6_12_18/hnasnet](#g3_4___atrous_6_12_18_hnasnet_)
        - [g2       @ atrous:6_12_18/hnasnet](#g2___atrous_6_12_18_hnasnet_)
        - [g3       @ atrous:6_12_18/hnasnet](#g3___atrous_6_12_18_hnasnet_)
        - [g4       @ atrous:6_12_18/hnasnet](#g4___atrous_6_12_18_hnasnet_)

<!-- /MarkdownTOC -->

<a id="build_dat_a_"></a>
# build_data

<a id="g2_4___build_data_"></a>
## g2_4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2_4 create_raw_seg=0
<a id="g3_4___build_data_"></a>
## g3_4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3_4 create_raw_seg=0
<a id="g2___build_data_"></a>
## g2       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2 create_raw_seg=0
<a id="g3___build_data_"></a>
## g3       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g3 create_raw_seg=0
<a id="g4___build_data_"></a>
## g4       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g4 create_raw_seg=0

<a id="hnasnet_"></a>
# hnasnet

<a id="atrous_6_12_18___hnasne_t_"></a>
## atrous:6_12_18       @ hnasnet-->new_deeplab_ipsc

<a id="g2_4___atrous_6_12_18_hnasnet_"></a>
### g2_4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g2_4,_train_:b2 start=1

<a id="g3_4___atrous_6_12_18_hnasnet_"></a>
### g3_4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g3_4,_train_:b2 start=0

<a id="g2___atrous_6_12_18_hnasnet_"></a>
### g2       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g2,_train_:b2 start=0

<a id="g3___atrous_6_12_18_hnasnet_"></a>
### g3       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g3,_train_:b2 start=0

<a id="g4___atrous_6_12_18_hnasnet_"></a>
### g4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:vis:g4,_train_:b2 start=0
