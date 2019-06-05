This repository provides the code for all experiments reported in [this paper](https://arxiv.org/abs/1901.04412).
It contains modified versions of several open source repositories that were used for experimentation though not all of these were reported in the paper.
These are the reported models and their corresponding folders:
1. DenseNet: [densenet](https://github.com/abhineet123/river_ice_segmentation/tree/master/densenet)
2. DeepLab: [deeplab](https://github.com/abhineet123/river_ice_segmentation/tree/master/deeplab)
3. UNet, SegNet: [image-segmentation-keras](https://github.com/abhineet123/river_ice_segmentation/tree/master/image-segmentation-keras)
4. SVM: [svm](https://github.com/abhineet123/river_ice_segmentation/tree/master/svm)

Unreported models:

1. FCN: [image-segmentation-keras](https://github.com/abhineet123/river_ice_segmentation/tree/master/image-segmentation-keras)
2. Video Segmentation: [video](https://github.com/abhineet123/river_ice_segmentation/tree/master/video)


The commands for running each model are provided in a .md file in the corresponding folder. For example, commands for UNet and DenseNet are in [image-segmentation-keras/unet.md](https://github.com/abhineet123/river_ice_segmentation/blob/master/image-segmentation-keras/unet.md) and [densenet/densenet.md](https://github.com/abhineet123/river_ice_segmentation/blob/master/densenet/densenet.md).
The commands are organized hierarchically into categories of experiments and a table of contents is included for easier navigation.

Following scripts can be used for data preparation and results generation:

1. Data augmentation / sub patch generation: [subPatchDataset.py](https://github.com/abhineet123/river_ice_segmentation/blob/master/subPatchDataset.py), [subPatchBatch.py](https://github.com/abhineet123/river_ice_segmentation/blob/master/subPatchBatch.py)
2. Stitching sub patch segmentation results and optionally evaluate them: [stitchSubPatchDataset.py](https://github.com/abhineet123/river_ice_segmentation/blob/master/stitchSubPatchDataset.py)
3. Generate ice concentration plots: [plotIceConcentration.py](https://github.com/abhineet123/river_ice_segmentation/blob/master/plotIceConcentration.py)
4. Visualize and evaluate segmentation results: [visDataset.py](https://github.com/abhineet123/river_ice_segmentation/blob/master/visDataset.py)


Commands for running these are in [river_ice_segm.md](https://github.com/abhineet123/river_ice_segmentation/blob/master/river_ice_segm.md) as well as in the individual model files.

Some commands might require general utility scripts available in the [python tracking framework](https://github.com/abhineet123/PTF), e.g. [videoToImgSeq.py](https://github.com/abhineet123/PTF/blob/master/videoToImgSeq.py).

If a command does not work,  the command corresponding to some experiment cannot be found or the meaning of some command is not clear, please create an issue and we will do our best to address it.

All the accompanying data is available [here](https://ualbertaca-my.sharepoint.com/:f:/g/personal/asingh1_ualberta_ca/EtwQsFI1rCRPm8kE7yv1p8IBCBBBh_vT9RYRIqrfDjXTHQ).
All commands assume that the data is present under _/data/617/_.



The code and data are released under [BSD license](https://opensource.org/licenses/BSD-3-Clause) and are free for research and commercial applications. 
Also, individual repositories used here might have their own licenses that might be more restrictive so please refer to them as well.

If you find this work useful, please consider citing [this paper](https://arxiv.org/abs/1901.04412) [[bibtex](https://github.com/abhineet123/river_ice_segmentation/blob/master/bibtex.txt)].






