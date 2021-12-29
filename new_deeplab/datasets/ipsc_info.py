def irange(a, b):
    return list(range(a, b + 1))


class IPSCInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 19)
            self.g1 = irange(0, 2)
            self.g2 = irange(3, 5)
            self.g3 = irange(6, 13)
            self.g4 = irange(14, 19)
            self.g4s = irange(14, 18)
            self.test = [20, ]

            self.g2_4 = self.g2 + self.g3 + self.g4
            self.g3_4 = self.g3 + self.g4
            self.g3_4s = self.g3 + self.g4s

    sequences = {
        # g1
        0: ('Frame_101_150_roi_7777_10249_10111_13349', 6),
        1: ('Frame_101_150_roi_12660_17981_16026_20081', 3),
        2: ('Frame_101_150_roi_14527_18416_16361_19582', 3),
        # g2
        3: ('Frame_150_200_roi_7644_10549_9778_13216', 46),
        4: ('Frame_150_200_roi_9861_9849_12861_11516', 47),
        5: ('Frame_150_200_roi_12994_10915_15494_12548', 37),
        # g3
        6: ('Frame_201_250_roi_7711_10716_9778_13082', 50),
        7: ('Frame_201_250_roi_8094_13016_11228_15282', 50),
        8: ('Frame_201_250_roi_10127_9782_12527_11782', 50),
        9: ('Frame_201_250_roi_11927_12517_15394_15550', 48),
        10: ('Frame_201_250_roi_12461_17182_15894_20449_1', 30),
        11: ('Frame_201_250_roi_12527_11015_14493_12615', 50),
        12: ('Frame_201_250_roi_12794_8282_14661_10116', 50),
        13: ('Frame_201_250_roi_16493_11083_18493_12549', 49),
        # g4
        14: ('Frame_251__roi_7578_10616_9878_13149', 25),
        15: ('Frame_251__roi_10161_9883_13561_12050', 24),
        16: ('Frame_251__roi_12094_17082_16427_20915', 25),
        17: ('Frame_251__roi_12161_12649_15695_15449', 25),
        18: ('Frame_251__roi_12827_8249_14594_9816', 25),
        19: ('Frame_251__roi_16627_11116_18727_12582', 25),
        # test
        20: ('Test_211208', 59),
    }


class IPSCPatchesInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 19)
            self.g1 = irange(0, 2)
            self.g2 = irange(3, 5)
            self.g3 = irange(6, 13)
            self.g4 = irange(14, 19)
            self.g4s = irange(14, 18)
            self.test = [20, ]

            self.g2_4 = self.g2 + self.g3 + self.g4
            self.g3_4 = self.g3 + self.g4

    sequences = {
        # g1
        0: ('Frame_101_150_roi_7777_10249_10111_13349', 250),
        1: ('Frame_101_150_roi_12660_17981_16026_20081', 352),
        2: ('Frame_101_150_roi_14527_18416_16361_19582', 192),
        # g2
        3: ('Frame_150_200_roi_7644_10549_9778_13216', 269),
        4: ('Frame_150_200_roi_9861_9849_12861_11516', 446),
        5: ('Frame_150_200_roi_12994_10915_15494_12548', 233),
        # g3
        6: ('Frame_201_250_roi_7711_10716_9778_13082', 351),
        7: ('Frame_201_250_roi_8094_13016_11228_15282', 350),
        8: ('Frame_201_250_roi_10127_9782_12527_11782', 274),
        9: ('Frame_201_250_roi_11927_12517_15394_15550', 725),
        10: ('Frame_201_250_roi_12461_17182_15894_20449_1', 345),
        11: ('Frame_201_250_roi_12527_11015_14493_12615', 293),
        12: ('Frame_201_250_roi_12794_8282_14661_10116', 357),
        13: ('Frame_201_250_roi_16493_11083_18493_12549', 257),
        # g4
        14: ('Frame_251__roi_7578_10616_9878_13149', 125),
        15: ('Frame_251__roi_10161_9883_13561_12050', 143),
        16: ('Frame_251__roi_12094_17082_16427_20915', 150),
        17: ('Frame_251__roi_12161_12649_15695_15449', 225),
        18: ('Frame_251__roi_12827_8249_14594_9816', 100),
        19: ('Frame_251__roi_16627_11116_18727_12582', 125),
        # test
        20: ('Test_211208', 59),
    }
