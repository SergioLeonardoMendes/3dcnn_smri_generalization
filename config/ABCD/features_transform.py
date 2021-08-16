import tensorflow as tf

def features_transform(example):
    """ Receive and transform the features from features_dictionary.json"""

    # assign subjectid key
    example['info/subjectid'] = example['info/subjectkey']

    # create interview_age_years
    example['info/interview_age_years'] = tf.cast(example['info/interview_age'], tf.float32) / 12.0

    return example
