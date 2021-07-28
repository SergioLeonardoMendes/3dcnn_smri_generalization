import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow_dataset import get_assess_dataset, get_dataset_iterator
from metrics import metrics_regression, metrics_classification
from utils import *

def predict(args, examples_to_predict, metrics_out_filepath, cutoff_value=None):
    examples_to_predict = pd.read_csv(examples_to_predict).values
    ids_output, labels_output, predictions_output = [], [], []
    df_metrics_reg, df_metrics_class = pd.DataFrame(), pd.DataFrame()
    # generate predictions for each of the models
    for output_name, loss in eval(args.model_losses).items():
        # load the saved model
        model = load_model(args.out_path + 'checkpoints/best_model')
        # assign the dataset and iterator
        dataset = get_assess_dataset(args, examples_to_predict)
        iterator = get_dataset_iterator(dataset, args.batch_size)
        # begin the predictions
        has_elements = True
        i = 1
        while has_elements:
            try:
                # get one batch
                inputs_it, labels_it = next(iterator)
                # make the predictions
                print_file(filename=args.log_file, text='  Executing predictions for ' + output_name + '... ' + str(i))
                predictions_it = model.predict(inputs_it)
                # store predictions of each class
                ids_output.extend(inputs_it['id'])
                predictions_output.extend(np.float32(predictions_it[:, 0]))
                labels_output.extend(labels_it[output_name])
                i += 1
            except StopIteration:
                has_elements = False
                pass
        # convert lists to np.array
        labels_output = np.array(labels_output)
        predictions_output = np.array(predictions_output)

        # print metrics
        if loss == 'MSE' or loss == 'MAE':
            # pearson_coef, _ = stats.pearsonr(labels_output, predictions_output)
            # print_file('\n##### Correlation without correction (linear fit): ' + str(pearson_coef), metrics_out_filepath)
            # # Linear correction for the predicted output
            # z = np.polyfit(labels_output, predictions_output, 1)
            # predictions_output_corrected = labels_output + predictions_output - (z[1] + z[0] * labels_output)
            # pearson_coef, _ = stats.pearsonr(labels_output, predictions_output_corrected)
            # print_file('##### Correlation with correction (linear fit): ' + str(pearson_coef), metrics_out_filepath)
            df_metrics_reg = df_metrics_reg.append(
                metrics_regression(name_output=eval(args.model_outputs)[output_name],
                                   ids_output=ids_output,
                                   labels_output=labels_output,
                                   predictions_output=predictions_output,
                                   results_filepath=metrics_out_filepath))
            # print(df_metrics_reg)
        elif loss == 'binary_crossentropy':
            if cutoff_value is None:
                cutoff_value = 0.5
            df_metrics_class = df_metrics_class.append(
                metrics_classification(name_output=eval(args.model_outputs)[output_name],
                                       ids_output=ids_output,
                                       labels_output=labels_output,
                                       predictions_output=predictions_output,
                                       cutoff_value=cutoff_value,
                                       results_filepath=metrics_out_filepath))

    return df_metrics_reg, df_metrics_class
