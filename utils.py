from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
import random
import json
import argparse
from datetime import datetime as dt
import os
import sys

class ArgumentParserNoExit(argparse.ArgumentParser):
    def error(self, message):
        pass

def set_config_generate_splits():
    """
    Set and return configuration parameters

    :return: configuration parameters
    """

    # read the project specific source path
    partial_parser = ArgumentParserNoExit(add_help=False)
    partial_parser.add_argument('--project_config_path', default='/path/to/dataset/configuration/')
    partial_arguments = partial_parser.parse_args()

    # load json parameters from dataset specific file
    json_file = open(partial_arguments.project_config_path + "default_parameters.json")
    config = json.load(json_file)
    json_file.close()

    # set default parameters
    config['stratify_config'] = "{'categoryA':0, 'categoryB':0, 'continuousA': 10}"

    config['out_path_splits'] = config['base_train_valid_test_path'] + 'kfold' + str(config['kfold']) + '-'\
                                + 'categoryA0categoryB0continuousA10' + '-seed' + str(config['seed']) + '/'

    # arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_config_path', type=str, default='/path/to/dataset/configuration/',
                        help="Dataset configuration path .")
    parser.add_argument("--base_train_valid_test_path", type=str, default=config['base_train_valid_test_path'],
                        help="Base filepath to generate train/valid/test data splits.")
    parser.add_argument("--csv_phenotypes", type=str, default=config['csv_phenotypes'],
                        help="Filepath to csv phenotypes.")
    parser.add_argument("--kfold", type=int, default=config['kfold'],
                        help="Number of k in kfold cross-validation.")
    parser.add_argument("--seed", type=int, default=config['seed'],
                        help="Random seed to use.")
    parser.add_argument("--out_path_splits", type=str, default=config['out_path_splits'],
                        help="Output path to store csv generated splits.")
    parser.add_argument("--stratify_config", type=str, default=config['stratify_config'],
                        help="Stratification hierarchical scheme. "
                             "Use zero for categories and quantiles_number for continuous variables split.")
    arguments = parser.parse_args()

    # compose final output path based on user inputs
    if arguments.out_path_splits == config['out_path_splits']:
        # create stratification string to identify destination path
        stratify_str = ''
        for column, qtl_num in eval(arguments.stratify_config).items():
            stratify_str = stratify_str + column + str(qtl_num)
        # assign final output path to data splits
        arguments.out_path_splits = config['base_train_valid_test_path'] + 'kfold' + str(arguments.kfold) + '-' \
                                    + stratify_str + '-seed' + str(arguments.seed) + '/'

    return arguments

def set_config_run_experiment():
    """
    Set and return configuration parameters

    :return: configuration parameters
    """

    # read the project specific source path
    partial_parser = ArgumentParserNoExit(add_help=False)
    partial_parser.add_argument('--project_config_path', default='/path/to/dataset/configuration/')
    partial_arguments = partial_parser.parse_args()

    # add specific dataset source files to path
    sys.path.append(os.path.expanduser(partial_arguments.project_config_path))

    # load json parameters from dataset specific file
    json_file = open(partial_arguments.project_config_path + "default_parameters.json")
    config = json.load(json_file)
    json_file.close()

    # set default parameters
    config['model_name'] = config['model_prefix']
    config['out_path_tmp'] = config['out_path'] + config['model_name'] + "/"
    config['log_file'] = config['out_path_tmp'] + "output.log"
    config['tboard_path'] = config['out_path_tmp'] + "tboard/"

    # arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_config_path', type=str, default='/path/to/dataset/configuration/',
                        help="Dataset configuration path .")
    parser.add_argument("--train_valid_test_path", type=str, default='/path/to/train_valid_test_data_splits/',
                        help="Base filepath to read Kfold train/valid/test data splits.")
    parser.add_argument("--n_folds_run", type=int, default=config['n_folds_run'],
                        help="Number of K folds to train on Kfold data splits.")
    parser.add_argument("--model_prefix", type=str, default=config['model_prefix'],
                        help="Base model name prefix.")
    parser.add_argument("--model_name", type=str, default=config['model_name'],
                        help="Base model name (override model_prefix if it is specified).")
    parser.add_argument("--data_path", type=str, default=config['data_path'],
                        help="Data path read files.")
    parser.add_argument("--out_path", type=str, default=config['out_path_tmp'],
                        help="Output path to save files.")
    parser.add_argument("--csv_phenotypes", type=str, default=config['csv_phenotypes'],
                        help="Filepath to csv phenotypes.")
    parser.add_argument("--log_file", type=str, default=config['log_file'],
                        help="Full filename to log output.")
    parser.add_argument("--bg_mask", type=str, default=config['bg_mask'],
                        help="Background mask of brain images.")
    parser.add_argument("--cache", type=int, default=config['cache'],
                        help="Cache dataset to ram (0=no, 1=yes).")
    parser.add_argument("--seed", type=int, default=config['seed'],
                        help="Random seed to use.")
    parser.add_argument("--n_epochs", type=int, default=config['n_epochs'],
                        help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=config['batch_size'],
                        help="Number of examples per training batch.")
    parser.add_argument("--lr", type=float, default=config['lr'],
                        help="Learning rate.")
    parser.add_argument("--beta_1", type=float, default=config['beta_1'],
                        help="Adam optimizer beta_1.")
    parser.add_argument("--beta_2", type=float, default=config['beta_2'],
                        help="Adam optimizer beta_2.")
    parser.add_argument("--kern_reg_l2", type=float, default=config['kern_reg_l2'],
                        help="Model kernel regularizer l2.")
    parser.add_argument("--dropout", type=float, default=config['dropout'],
                        help="Dropout layer values.")
    parser.add_argument("--augment", type=int, default=config['augment'],
                        help="Training image augmentation (0=no, 1=yes)")
    parser.add_argument("--model_arch", type=str, default=config['model_arch'],
                        help="Model architecture.")
    parser.add_argument("--model_outputs", type=str, default=config['model_outputs'],
                        help="Model output keys.")
    parser.add_argument("--model_losses", type=str, default=config['model_losses'],
                        help="Model output optimization losses.")
    parser.add_argument("--model_out_bias", type=str, default=config['model_out_bias'],
                        help="Model bias initialization for output layers.")
    parser.add_argument("--model_metrics", type=str, default=config['model_metrics'],
                        help="Model optimization metrics.")
    arguments = parser.parse_args()

    # compose final arguments based on user inputs
    if arguments.model_name == config['model_name']:
        # solve the case where the last character is a slash
        if arguments.train_valid_test_path[-1:] == '/':
            str_fin_pos = arguments.train_valid_test_path.rfind('/')
            str_ini_pos = arguments.train_valid_test_path[:str_fin_pos - 1].rfind('/') + 1
        else:
            str_fin_pos = len(arguments.train_valid_test_path)
            str_ini_pos = arguments.train_valid_test_path.rfind('/') + 1
        str_stratify_schema = arguments.train_valid_test_path[str_ini_pos:str_fin_pos]
        arguments.model_name = arguments.model_prefix + str_stratify_schema
    if arguments.out_path == config['out_path_tmp']:
        arguments.out_path = config['out_path'] + arguments.model_name
    if arguments.log_file == config['log_file']:
        arguments.log_file = arguments.out_path + '/' + "output.log"

    return arguments

def set_config_evaluate_cross_dataset():
    """
    Set and return configuration parameters

    :return: configuration parameters
    """

    # read the project specific source path
    partial_parser = ArgumentParserNoExit(add_help=False)
    partial_parser.add_argument('--saved_model_path', default='/root_path/of/saved_model/')
    partial_parser.add_argument('--full_dataset_csv_path', default='/full_dataset/to/filename.csv')
    partial_parser.add_argument('--config_model_path', default='/path/to/model/configuration/')
    partial_parser.add_argument('--config_dataset_path', default='/path/to/dataset/configuration/')
    partial_arguments = partial_parser.parse_args()

    # add specific dataset source files to path
    sys.path.append(os.path.expanduser(partial_arguments.config_dataset_path))

    # load json parameters from model and dataset specific files
    json_file = open(partial_arguments.config_model_path + "default_parameters.json")
    config_model = json.load(json_file)
    json_file.close()
    json_file = open(partial_arguments.config_dataset_path + "default_parameters.json")
    config_dataset = json.load(json_file)
    json_file.close()

    # set default parameters
    config = dict()

    # from dataset
    str_fin_pos = config_dataset['data_path'].rfind('/')
    str_ini_pos = config_dataset['data_path'][:str_fin_pos - 1].rfind('/') + 1
    config['dataset_name'] = config_dataset['data_path'][str_ini_pos:str_fin_pos]
    config['project_config_path'] = partial_arguments.config_dataset_path
    config['model_outputs'] = config_dataset['model_outputs']
    config['model_losses'] = config_dataset['model_losses']
    config['model_metrics'] = config_dataset['model_metrics']
    # from model
    config['bg_mask'] = config_model['bg_mask']
    config['cache'] = config_model['cache']
    config['batch_size'] = config_model['batch_size']
    config['augment'] = config_model['augment']
    config['seed'] = config_model['seed']
    # generated based on input values
    config['saved_model_path'] = partial_arguments.saved_model_path
    if config['saved_model_path'][-1:] != '/':
        config['saved_model_path'] = config['saved_model_path'] + '/'
    str_fin_pos = config['saved_model_path'].rfind('/')
    str_ini_pos = config['saved_model_path'][:str_fin_pos - 1].rfind('/') + 1
    config['model_name'] = config['saved_model_path'][str_ini_pos:str_fin_pos]
    config['full_dataset_csv_path'] = partial_arguments.full_dataset_csv_path
    config['out_path'] = config['saved_model_path']
    config['log_file'] = config['saved_model_path'][:-4] + 'results/' + config['saved_model_path'][-4:] \
                         + config['dataset_name'] + '/output.log'
    config['result_metrics_path'] = config['saved_model_path'][:-4] + 'results/' + config['saved_model_path'][-4:] \
                                    + config['dataset_name'] + '/'

    # arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_config_path', type=str, default=config['project_config_path'],
                        help="Dataset configuration path.")
    parser.add_argument('--config_model_path', default='/path/to/model/configuration/',
                         help="Path to model configuration files.")
    parser.add_argument('--config_dataset_path', default='/path/to/dataset/configuration/',
                         help="Path to dataset configuration files.")
    parser.add_argument('--dataset_name', type=str, default=config['dataset_name'],
                        help="Dataset name.")
    parser.add_argument("--full_dataset_csv_path", type=str, default=config['full_dataset_csv_path'],
                        help="Full dataset csv filepath.")
    parser.add_argument('--saved_model_path', type=str, default=config['saved_model_path'],
                        help="Base path of the saved model.")
    parser.add_argument("--model_name", type=str, default=config['model_name'],
                        help="Saved model name.")
    parser.add_argument("--out_path", type=str, default=config['out_path'],
                        help="Output path to save files.")
    parser.add_argument("--log_file", type=str, default=config['log_file'],
                        help="Full filename to log output.")
    parser.add_argument("--result_metrics_path", type=str, default=config['result_metrics_path'],
                        help="Path to save the resultant metrics.")
    parser.add_argument("--bg_mask", type=str, default=config['bg_mask'],
                        help="Background mask of brain images.")
    parser.add_argument("--cache", type=int, default=config['cache'],
                        help="Cache dataset to ram (0=no, 1=yes).")
    parser.add_argument("--seed", type=int, default=config['seed'],
                        help="Random seed to use.")
    parser.add_argument("--batch_size", type=int, default=config['batch_size'],
                        help="Number of examples per training batch.")
    parser.add_argument("--augment", type=int, default=config['augment'],
                        help="Training image augmentation (0=no, 1=yes)")
    parser.add_argument("--model_losses", type=str, default=config['model_losses'],
                        help="Model output optimization losses.")
    parser.add_argument("--model_metrics", type=str, default=config['model_metrics'],
                        help="Model optimization metrics.")
    parser.add_argument("--model_outputs", type=str, default=config['model_outputs'],
                        help="Model output keys.")
    arguments = parser.parse_args()

    return arguments


def print_file(text, filename=""):
    """
    Print text to screen and log_file

    :param filename: filename to print text
    :param text: string to print (accepts formatting)
    """

    # open and write log_file
    f = open(filename, "a")

    # if text is empty, log the current time
    if text == "":
        text = dt.now().strftime("%H:%M:%S,%f ")

    f.write(text+"\n")
    f.close()
    print(text)

def create_dirs(args):
    try:
        print(args.out_path)
        os.makedirs(args.out_path)
        print(args.tboard_path)
        os.makedirs(args.tboard_path)
    except OSError:
        print("\nCreation of directories failed.")
    else:
        print_file(filename=args.log_file, text="\nSuccessfully created directories.")

def print_configs(args):
    """
    Print execution parameters
    """
    print_file(filename=args.log_file, text="\n----- Execution parameters -----")
    for k, v in vars(args).items():
        print_file(filename=args.log_file, text=f"  {k}: {v}")

def config_gpus(args):
    """
    Configure gpus strategy
    """
    # print_file(filename=args.log_file, text='\n----- Enable TF_XLA_FLAGS -----')
    # os.environ['TF_XLA_FLAGS'] = "--tf_xla_enable_xla_devices" #"--tf_xla_auto_jit=2"

    print_file(filename=args.log_file, text='\n----- Enable Mixed precision -----')
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    print_file(filename=args.log_file, text='  Compute dtype: %s' % policy.compute_dtype)
    print_file(filename=args.log_file, text='  Variable dtype: %s' % policy.variable_dtype)
    tf.keras.mixed_precision.set_global_policy(policy)

    local_device_proto = device_lib.list_local_devices()
    all_devices = [x.name for x in local_device_proto]
    n_gpus = len(tf.config.list_physical_devices('GPU'))
    print_file(filename=args.log_file, text="\n----- GPUs configuration -----")
    print_file(filename=args.log_file, text="  Devices: " + str(all_devices))
    print_file(filename=args.log_file, text="  Number of GPUs: " + str(n_gpus))

def enable_determinism(args):
    """
    Set seeds to enable deterministic operations
    """
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print_file(filename=args.log_file, text="\n----- Enable TF Deterministic -----")
