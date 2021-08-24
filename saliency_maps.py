from tensorflow.keras.models import load_model
from tensorflow.keras import backend as k
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import numpy as np
import pandas as pd
from predict import get_assess_dataset, get_dataset_iterator

SMOOTH_S = 5
SMOOTH_N = 10

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
                                    smooth_samples=SMOOTH_S, smooth_noise=SMOOTH_N)
            # append gradients and images to lists
            grad_list.extend(saliency_map)
            image_list.extend(x)
            i += 1
        except StopIteration:
            has_elements = False
            pass
    saliency_mean = np.mean(grad_list, axis=0)[:, :, :]
    brain_mean = np.mean(image_list, axis=0)[:, :, :]

    return saliency_mean, brain_mean
