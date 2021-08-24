import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt
from utils import print_file

def sens_spec_h_mean(labels_output, predictions_bin):
    """ Calculate sensitivity, specificity and harmonic mean between both of them """
    tn, fp, fn, tp = metrics.confusion_matrix(labels_output, predictions_bin).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    h_mean = 2 * sens * spec / (sens + spec)
    return sens, spec, h_mean


def metrics_classification(name_output, ids_output, labels_output, predictions_output, cutoff_value, results_filepath):
    log_output = results_filepath

    # calculate metrics
    predictions_bin = predictions_output >= cutoff_value
    sens, spec, h_mean = sens_spec_h_mean(labels_output, predictions_bin)
    num_examples = len(predictions_output)
    num_class0 = len(labels_output) - np.sum(labels_output)
    num_class1 = np.sum(labels_output)
    auc_roc = metrics.roc_auc_score(labels_output, predictions_output)
    simple_acc = metrics.accuracy_score(labels_output, predictions_bin)
    prec = metrics.precision_score(labels_output, predictions_bin)
    rec = metrics.recall_score(labels_output, predictions_bin)
    f1scr = metrics.f1_score(labels_output, predictions_bin)

    # print metrics
    print_file('\n\n*************************************************************', log_output)
    print_file('  Output = ' + name_output, log_output)
    print_file('*************************************************************\n', log_output)
    print_file('  Number of examples: ' + str(num_examples), log_output)
    print_file('  Class=1 examples:' + str(num_class1), log_output)
    print_file('  Class=0 examples:' + str(num_class0) + '\n', log_output)
    print_file('  AUC/ROC: ' + str(auc_roc), log_output)
    print_file('  Simple Accuracy: ' + str(simple_acc), log_output)
    print_file('  Precision=[true_positives/(true_positives+false_positives)]: ' + str(prec), log_output)
    print_file('  Recall=[true_positives/(true_positives+false_negatives)]: ' + str(rec), log_output)
    print_file('  F1 Score=[2*precision*recall/(precision+recall)]: ' + str(f1scr)+'\n', log_output)
    # print detailed predictions
    print_file('\n--- Detailed Predictions:', log_output)
    print_file('\nsubjectid,target,predicted', log_output)
    for i in range(num_examples):
        print_file(str(ids_output[i]) + ','
                   + str(np.round(labels_output[i], 1)) + ','
                   + str(np.round(predictions_output[i], 1)), log_output)

    # search for the best cutoff and assign it to best_cutoff
    best_h_mean, best_cutoff = 0, 0
    for cutoff_try in np.arange(0, 1, 0.02):
        predictions_bin_try = predictions_output >= cutoff_try
        sens_try, spec_try, h_mean_try = sens_spec_h_mean(labels_output, predictions_bin_try)
        # f1_scr = metrics.f1_score(labels_output, predictions_bin_try)
        if h_mean_try > best_h_mean:
            best_h_mean = h_mean_try
            best_cutoff = cutoff_try

    metric_fields = {'output': [name_output],
                     'num_examples': [num_examples],
                     'num_class0': [num_class0],
                     'num_class1': [num_class1],
                     'cutoff_val': [cutoff_value],
                     'cutoff_ops': [best_cutoff],
                     'auc_roc': [auc_roc],
                     'simple_acc': [simple_acc],
                     'f1scr': [f1scr],
                     'h_mean': [h_mean],
                     'prec': [prec],
                     'spec': [spec],
                     'rec': [rec],
                     'sens': [sens]
                     }
    df_tmp_metrics = pd.DataFrame(data=metric_fields)

    return df_tmp_metrics

def metrics_regression(name_output, ids_output, labels_output, predictions_output, results_filepath):
    """ Calculate and return regression metrics """

    log_output = results_filepath
    results_prefix = results_filepath[:results_filepath.rfind('.')] + '_'

    # calculate metrics
    num_examples = len(predictions_output)
    lbl_mean = np.mean(labels_output)
    lbl_median = np.median(labels_output)
    lbl_std = np.std(labels_output)
    pred_mean = np.mean(predictions_output)
    pred_median = np.median(predictions_output)
    pred_std = np.std(predictions_output)
    mae = np.mean(np.abs(predictions_output - labels_output))
    r2q2 = metrics.r2_score(labels_output, predictions_output)
    pearson_coef, p_value = stats.pearsonr(labels_output, predictions_output)

    # print metrics
    print_file('\n\n*************************************************************', log_output)
    print_file('  Output = ' + name_output, log_output)
    print_file('*************************************************************\n', log_output)
    print_file('  Number of examples: ' + str(num_examples) + '\n', log_output)
    print_file('  --------- Labels distribution ---------', log_output)
    print_file('  Mean: ' + str(lbl_mean), log_output)
    print_file('  Median: ' + str(lbl_median), log_output)
    print_file('  Standard deviation: ' + str(lbl_std), log_output)
    print_file('\n  --------- Predictions distribution ---------', log_output)
    print_file('  Mean: ' + str(pred_mean), log_output)
    print_file('  Median: ' + str(pred_median), log_output)
    print_file('  Standard Deviation: ' + str(pred_std), log_output)
    print_file('\n  --- Mean Absolute Error ---', log_output)
    print_file('  MAE: ' + str(mae), log_output)
    print_file('\n  ----- R2 Score-----', log_output)
    print_file('  R2 SkLearn: ' + str(r2q2) + '\n', log_output)
    print_file('  --- Pearson Correlation (predicted vs correct) ---', log_output)
    print_file('  Correlation: ' + str(pearson_coef) + '\n', log_output)
    print_file('  P-value: ' + str(p_value), log_output)
    # print detailed predictions
    print_file('\n--- Detailed Predictions:', log_output)
    print_file('\nsubjectid,target,predicted,deviation', log_output)
    for i in range(len(labels_output)):
        print_file(str(ids_output[i]) + ', '
                   + str(np.round(labels_output[i], 1)) + ','
                   + str(np.round(predictions_output[i], 1)) + ','
                   + str(np.round(predictions_output[i] - labels_output[i], 1)), log_output)

    # Scatter Plot
    plt.figure(10)
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    _ = ax.scatter(labels_output, predictions_output)
    plt.title('Scatter Plot')
    plt.xlabel('T A R G E T')
    plt.ylabel('P R E D I C T I O N')
    x = labels_output
    y = predictions_output
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_axe = np.linspace(x.min(), x.max(), 3000)
    y_axe = np.linspace(p(x).min(), p(x).max(), 3000)
    ax.plot(x_axe, y_axe, "blue", ls='-', lw=2, label='Model linear fit')
    ax.plot(x_axe, x_axe, "limegreen", ls='-', lw=2, label='Ideal linear fit')
    ax.legend(bbox_to_anchor=(0, 0.88), loc='upper left')
    text_str = ('$\it{n=' + str(num_examples) +
                ', MAE='+str(np.round(mae, 2)) +
                ', r=' + str(np.round(pearson_coef, 2)) + '}$')
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5)
    ax.text(0.027, 0.96, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    plt.savefig(results_prefix + 'predictions_scatter.png')

    # Plot historams
    plt.figure(12)
    plt.clf()
    plt.cla()
    plt.title('Histogram - Correct Targets')
    _ = plt.hist(labels_output)
    plt.savefig(results_prefix + 'targets_histogram.png')
    plt.figure(13)
    plt.clf()
    plt.cla()
    plt.title('Histogram - Predictions')
    _ = plt.hist(predictions_output)
    plt.savefig(results_prefix + 'predictions_histogram.png')

    metric_fields = {'output': [name_output],
                     'num_examples': [num_examples],
                     'lbl_mean': [lbl_mean],
                     'lbl_median': [lbl_median],
                     'lbl_std': [lbl_std],
                     'pred_mean': [pred_mean],
                     'pred_median': [pred_median],
                     'pred_std': [pred_std],
                     'mae': [mae],
                     'r2q2': [r2q2],
                     'r_pearson': [pearson_coef],
                     'r_pvalue': [p_value]
                     }
    df_tmp_metrics = pd.DataFrame(data=metric_fields)

    return df_tmp_metrics
