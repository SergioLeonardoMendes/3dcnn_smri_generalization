import numpy as np

def features_transform(example):
    """ Receive and transform the features from features_dictionary.json"""
    # map control (label=2 to label=0) woman=0, man=1
    if example['info/gender'] == 2:
        example['info/gender'] = np.int64(0)

    # map control (label=2 to label=1) dcany = any psychopathological condition
    if example['info/dcany'] == 2.0:
        example['info/dcany'] = 1.0

    return example
