import tensorflow as tf
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import json

# CONSTANT DECLARATIONS
IMAGE_SIZE_IN = [121, 145, 121, 3]

def fit_background_mask():
    """
    Fit image background mask to 128x128x128x3

    :return: resultant background mask of 128x128x128x3
    """
    # load the image background mask and pad it to the same format of training images
    paddings = tf.constant([[3, 4, ], [0, 0], [3, 4], [0, 0]])
    result_mask4d = np.int64(np.load(ARGS.bg_mask))
    result_mask4d = tf.pad(result_mask4d[:, 11:-6, :, :], paddings, "CONSTANT", constant_values=1)
    result_mask4d = tf.cast(result_mask4d, tf.float32)
    return result_mask4d

def load_dataset(filenames):
    # Read from TFRecords. For optimal performance, we interleave reads from multiple files.
    records = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    return records.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

def get_assess_dataset(arguments, filenames):
    global ARGS
    ARGS = arguments
    if ARGS.cache == 1:
        result = load_dataset(filenames).cache().batch(ARGS.batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        result = load_dataset(filenames).batch(ARGS.batch_size).prefetch(tf.data.AUTOTUNE)
    return result

@tf.function
def augmentation(volume, seed):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume, seed):
        # generate a random stateless rotation angle index
        angles = np.arange(-40, 41, 1)
        prob_angles = np.ones(len(angles))
        index_angle = tf.random.stateless_categorical(logits=tf.math.log([prob_angles]),
                                                      num_samples=1, seed=seed)[0, 0].numpy()
        # generate a random stateless axe index
        axes = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        prob_axes = np.ones(len(axes))
        index_axe = tf.random.stateless_categorical(logits=tf.math.log([prob_axes]),
                                                    num_samples=1, seed=seed)[0, 0].numpy()
        # rotate volume
        cp_volume = cp.asarray(volume)
        volume = cp.asnumpy(ndimage.rotate(cp_volume, angle=angles[index_angle],
                                           axes=axes[index_axe], reshape=False))
        del cp_volume

        return volume

    def scipy_shift(volume, seed):
        # generate a random stateless pixel index
        pixels = np.arange(-10, 11, 1)
        prob_pixels = np.ones(len(pixels))
        index_pixel = tf.random.stateless_categorical(logits=tf.math.log([prob_pixels]),
                                                      num_samples=1, seed=seed)[0, 0].numpy()
        # generate a random stateless axe index
        axes = [0, 1, 2]
        prob_axes = np.ones(len(axes))
        index_axe = tf.random.stateless_categorical(logits=tf.math.log([prob_axes]),
                                                    num_samples=1, seed=seed)[0, 0].numpy()
        # assign a random pixel value to a random axe
        shift_values = [0, 0, 0, 0]
        shift_values[index_axe] = pixels[index_pixel]
        # shift volume
        cp_volume = cp.asarray(volume)
        volume = cp.asnumpy(ndimage.shift(cp_volume, shift=shift_values))
        del cp_volume

        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume, seed], tf.float32)
    augmented_volume = tf.numpy_function(scipy_shift, [augmented_volume, seed], tf.float32)
    return augmented_volume

def train_augment(dataset, example_seed):
    """Process training data by rotating. """
    # Rotate volume
    # seed = RNG.make_seeds(2)[0]
    (volume, subjectid), label = dataset
    volume = augmentation(volume['image'], seed=example_seed)
    return ({'image': volume, 'id': subjectid}), label

def get_training_dataset(arguments, filenames):
    global ARGS
    ARGS = arguments
    train_fns = filenames
    train_len = len(train_fns)
    if ARGS.cache == 1:
        dataset = load_dataset(train_fns).cache().repeat().shuffle(train_len)
    else:
        dataset = load_dataset(train_fns).repeat()
    if ARGS.augment == 1:
        seeds = tf.data.experimental.RandomDataset(seed=ARGS.seed).batch(2)
        dataset = tf.data.Dataset.zip((dataset, seeds))
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = dataset.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    result = dataset.batch(ARGS.batch_size).prefetch(tf.data.AUTOTUNE)

    return result

# def get_training_dataset_balanced(args, train_fns_pos, train_fns_neg):
#     train_len_pos = len(train_fns_pos)
#     train_len_neg = len(train_fns_neg)
#
#     if args.cache == 1:
#         dataset_pos = load_dataset(train_fns_pos).cache().shuffle(train_len_pos).repeat()
#         dataset_neg = load_dataset(train_fns_neg).cache().shuffle(train_len_neg)
#     else:
#         dataset_pos = load_dataset(train_fns_pos).repeat()
#         dataset_neg = load_dataset(train_fns_neg)
#
#     resampled_ds = tf.data.Dataset.zip((dataset_pos, dataset_neg))
#     resampled_ds = resampled_ds.flat_map(
#         lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
#             tf.data.Dataset.from_tensors(ex_neg)))
#
#     result = resampled_ds.repeat().batch(ARGS.batch_size).prefetch(tf.data.AUTOTUNE)
#     return result #.shuffle(816)

def get_dataset_iterator(dataset, n_examples):
    return dataset.unbatch().batch(n_examples).as_numpy_iterator()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_image(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def parse_tfrecord(serialized_example):
    """ Decode examples stored in TFRecord files """
    # import dataset specific features_transform function
    from features_transform import features_transform

    # load json features from dataset specific features_dictionary.json
    json_file = open(ARGS.project_config_path + "features_dictionary.json")
    features_config = json.load(json_file)
    json_file.close()
    features = {}
    for key, value in features_config.items():
        features[key] = eval(value)

    # load features as describe in dataset specific features_dictionary.json
    example = tf.io.parse_single_example(serialized_example, features)

    # fit the image into a shape of 128x128x128x3
    paddings = tf.constant([[3, 4, ], [0, 0], [3, 4], [0, 0]])  # mask to pad
    # the parameter CONSTANT means to pad with 0's
    image_fitted = tf.pad(example['image'][:, 11:-6, :, :], paddings, "CONSTANT")

    # transform features as described in dataset specific features_transform.py
    example = features_transform(example)

    # get the image background mask and apply a mask composed by 0's
    background_mask4d = tf.cast((fit_background_mask != 1.0), tf.float32)
    image_fitted = image_fitted * background_mask4d
    # get the image background mask and apply a mask composed by -1's
    background_mask4d = fit_background_mask()
    image_fitted = image_fitted + (background_mask4d * -1)

    # remove csf layer
    image_fitted = image_fitted[:, :, :, :2]

    # prepare output targets according to user inputs
    output_dict = {}
    for key, value in eval(ARGS.model_outputs).items():
        output_dict[key] = example[value]

    # output_data = (({'image':image_fitted, 'sex':sex, 'age':age}),
    #                ({'cbcltot':cbcltot}))

    output_data = (({'image': image_fitted, 'id': example['info/subjectid']}),
                   output_dict)

    return output_data
