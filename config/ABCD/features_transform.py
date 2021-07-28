def features_transform(example):
    """ Receive and transform the features from features_dictionary.json"""

    # assign subjectid key
    example['info/subjectid'] = example['info/subjectkey']

    return example
