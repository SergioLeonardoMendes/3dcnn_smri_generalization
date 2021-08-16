import numpy as np

def features_transform(example):
    """ Receive and transform the features from features_dictionary.json"""

    # assign subjectid key
    example['info/subjectid'] = example['info/SUB_ID']

    # map health control (label=2 to label=0) DX_GROUP=2 is autism
    if example['info/DX_GROUP'] == 2:
        example['info/DX_GROUP'] = 0

    return example
