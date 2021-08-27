import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as k
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import SimpleITK as sitk
import numpy as np
import pandas as pd
from predict import get_assess_dataset, get_dataset_iterator

template_config_path = '/project/sources/config/'

def generate_saliency_maps(args, examples_to_map):
    # load examples filenames in pandas dataframe
    examples_to_map = pd.read_csv(examples_to_map).values
    # load the saved model
    model = load_model(args.out_path + 'checkpoints/best_model')
    # assign the dataset and iterator
    dataset = get_assess_dataset(args, examples_to_map)
    iterator = get_dataset_iterator(dataset, np.round(args.batch_size/5))
    # create tf-keras-vis objects
    replace2linear = ReplaceToLinear()
    saliency = Saliency(model,
                        model_modifier=replace2linear,
                        clone=True)
    # inactive_output = lambda output: K.mean(output * 0.0)
    active_output = lambda output: k.mean(output[:])
    # begin the saliency generation
    image_list, grad_list = [], []
    has_elements = True
    i = 1
    while has_elements:
        try:
            # get one batch
            inputs_it, labels_it = next(iterator)
            # assign the input image to X
            x = inputs_it['image']
            # Generate saliency map for the example
            saliency_map = saliency([active_output], x,
                                    smooth_samples=args.smoothgrad_sample,
                                    smooth_noise=args.smoothgrad_noise)
            # append gradients and images to lists
            grad_list.extend(saliency_map)
            image_list.extend(x)
            i += 1
        except StopIteration:
            has_elements = False
            pass
    # load brain mask and pad it to the same format of dataset images
    brain_mask4d = np.int64(np.load(args.bg_mask))
    paddings = tf.constant([[3, 4, ], [0, 0], [3, 4], [0, 0]])
    brain_mask4d = tf.pad(brain_mask4d[:, 11:-6, :, :], paddings, "CONSTANT", constant_values=1)
    # saliency and brain means
    saliency_mean = np.mean(grad_list, axis=0)[:, :, :]
    saliency_mean = saliency_mean * np.float32(tf.cast((brain_mask4d[:, :, :, 0] != 1), tf.float32))
    brain_mean = np.mean(image_list, axis=0)[:, :, :, :]
    brain_mean = brain_mean * np.float32(tf.cast((brain_mask4d[:, :, :, :2] != 1), tf.float32))
    # normalize saliency between 0-1
    saliency_mean_normalized = normalize_saliency_maps(saliency_mean)

    return saliency_mean_normalized, brain_mean

def normalize_saliency_maps(saliency_mean):
    normalized_saliency_mean = saliency_mean / np.max(saliency_mean)
    return normalized_saliency_mean

def save_saliency_brain_nii(result_metrics_path, saliency_mean, brain_mean):
    metadata_template = template_config_path + 'metadata_template_1-5mm.nii'
    # reshape images to 121x145x121 (voxel_size = 1.5 mm)
    paddings_saliency = tf.constant([[0, 0], [11, 6], [0, 0]])  # mask to pad
    paddings_brain = tf.constant([[0, 0], [11, 6], [0, 0], [0, 0]])  # mask to pad
    saliency_mean_reshaped = tf.pad(saliency_mean[3:-4, :, 3:-4],
                                    paddings_saliency, "CONSTANT")
    brain_mean_reshaped = tf.pad(brain_mean[3:-4, :, 3:-4, :],
                                 paddings_brain, "CONSTANT")
    # read metadata
    nii_metadata = sitk.ReadImage(metadata_template)
    # write attention mean to nifti file
    nii_saliency = sitk.GetImageFromArray(saliency_mean_reshaped)
    nii_saliency.CopyInformation(nii_metadata)
    sitk.WriteImage(nii_saliency, result_metrics_path + 'saliency_mean.nii')
    # write brain GM mean to nifti file
    nii_brain_gm = sitk.GetImageFromArray(brain_mean_reshaped[:, :, :, 0])
    nii_brain_gm.CopyInformation(nii_metadata)
    sitk.WriteImage(nii_brain_gm, result_metrics_path + 'brain_mean_gm.nii')
    # write brain WM mean to nifti file
    nii_brain_wm = sitk.GetImageFromArray(brain_mean_reshaped[:, :, :, 1])
    nii_brain_wm.CopyInformation(nii_metadata)
    sitk.WriteImage(nii_brain_wm, result_metrics_path + 'brain_mean_wm.nii')

def map_attention_rois(args):
    # aal template files
    aal3_template_file = template_config_path + 'AAL3v1_1-5mm.nii'
    aal3_labels_file = template_config_path + 'AAL3v1.nii.txt'
    # attention gradients file
    grads_file = args.out_path + '/results/saliency_mean.nii'
    # read attention gradients
    grads_nii = sitk.ReadImage(grads_file)
    attention_grads = sitk.GetArrayFromImage(grads_nii)
    # read aal3 data matrix
    aal3_nii = sitk.ReadImage(aal3_template_file)
    aal3_atlas = sitk.GetArrayFromImage(aal3_nii)
    # read aal3 labels
    df_atlas = pd.read_csv(aal3_labels_file, usecols=[0, 1], sep=' ', header=None, names=['id', 'name'])
    # calculate gradients' sum and average from each atlas roi
    row_list = []
    for cod, description in zip(df_atlas['id'].values, df_atlas['name'].values):
        mask = (aal3_atlas == cod)
        res_grads = attention_grads * mask
        v_sum = np.sum(res_grads)
        v_avg = (np.sum(res_grads) / np.sum(mask)) if (np.sum(mask) > 0) else 0.0
        grads_total = {'atlas_id': cod, 'atlas_descr': description, 'grads_sum': v_sum, 'grads_avg': v_avg}
        row_list.append(grads_total)
        # print(grads_total)
    # store values in pandas dataframe
    df_attention_rois = pd.DataFrame(row_list)
    # sort by gradients average
    df_attention_rois = df_attention_rois.sort_values(by=['grads_avg'], ascending=False)

    return df_attention_rois
