<!-- MarkdownTOC -->

- [build_data](#build_dat_a_)
    - [frame_150_end       @ build_data](#frame_150_end___build_data_)
- [hnasnet](#hnasnet_)
    - [atrous:6_12_18       @ hnasnet](#atrous_6_12_18___hnasne_t_)
        - [g2_4       @ atrous:6_12_18/hnasnet](#g2_4___atrous_6_12_18_hnasnet_)
        - [g3_4       @ atrous:6_12_18/hnasnet](#g3_4___atrous_6_12_18_hnasnet_)
        - [g4       @ atrous:6_12_18/hnasnet](#g4___atrous_6_12_18_hnasnet_)

<!-- /MarkdownTOC -->

<a id="build_dat_a_"></a>
# build_data

<a id="frame_150_end___build_data_"></a>
## frame_150_end       @ build_data-->new_deeplab_ipsc
python36 datasets/build_ipsc_data.py db_split=g2_4 create_raw_seg=1

<a id="hnasnet_"></a>
# hnasnet

<a id="atrous_6_12_18___hnasne_t_"></a>
## atrous:6_12_18       @ hnasnet-->new_deeplab_ipsc

<a id="g2_4___atrous_6_12_18_hnasnet_"></a>
### g2_4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g2_4,_train_:b2 start=0

<a id="g3_4___atrous_6_12_18_hnasnet_"></a>
### g3_4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ipsc_:train:g3_4,_train_:b2 start=0

<a id="g4___atrous_6_12_18_hnasnet_"></a>
### g4       @ atrous:6_12_18/hnasnet-->new_deeplab_ipsc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ipsc_:train:g4,_train_:b2 start=0
