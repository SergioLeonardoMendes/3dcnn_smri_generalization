import numpy as np

def features_transform(example):
    """ Receive and transform the features from features_dictionary.json"""

    # assign subjectid key
    example['info/subjectid'] = example['info/SCANDIR_ID']

    # create a combined diagnostic of ADHD (including: inattentive, hyperactive and combined)
    if example['info/DX'] == 0:
        example['info/DX_ALL'] = 0
    else:
        example['info/DX_ALL'] = 1

    return example
