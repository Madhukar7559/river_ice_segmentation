import os
import sys

sys.path.append('..')

import paramparse

import sys

sys.path.append('..')

from new_deeplab_train_params import NewDeeplabTrainParams
from new_deeplab_vis_params import NewDeeplabVisParams
#
# from visDataset import VisParams
# from stitchSubPatchDataset import StitchParams

import new_deeplab_train as train
import new_deeplab_vis as raw_vis

import stitchSubPatchDataset as stitch
import visDataset as vis


class Phases:
    train, raw_vis, stitch, vis = map(str, range(4))


class NewDeeplabParams:
    def __init__(self):
        self.cfg_root = 'cfg'
        self.cfg_ext = 'cfg'
        self.cfg = ()

        self.gpu = ""

        self.phases = '013'

        self.train = NewDeeplabTrainParams()
        self.raw_vis = NewDeeplabVisParams()
        self.stitch = stitch.StitchParams()
        self.vis = vis.VisParams()

    def process(self):
        self.train.process()
        self.raw_vis.process()
        self.stitch.process()
        self.vis.process()


def main():
    params = NewDeeplabParams()
    paramparse.process(params)

    params.process()

    if params.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    if Phases.train in params.phases:
        train.run(params.train)

    if Phases.raw_vis in params.phases:
        raw_vis.run(params.raw_vis)

    if Phases.stitch in params.phases:
        stitch.run(params.stitch)

    if Phases.vis in params.phases:
        vis.run(params.vis)


if __name__ == '__main__':
    main()
