from pathlib import Path
import copy
import pandas as pd
import train_fold
from predict import predict
from utils import *

def main(args):

    # solve the case where the last character is not a slash
    if args.train_valid_test_path[-1:] != '/':
        args.train_valid_test_path = args.train_valid_test_path + '/'

    # create an specific folder arguments variable
    args_fold = copy.copy(args)

    # create experiment main directory
    try:
        print('\nCreating directory: ' + args.out_path)
        os.makedirs(args.out_path, exist_ok=True)
    except OSError:
        print("Creation of directories failed.\n")
    else:
        print("Directory successfully created!\n")

    # remove output_log if it already exists
    os.remove(args.log_file) if os.path.exists(args.log_file) else None

    # print configs of the experiment
    print_configs(args)

    print_file(filename=args.log_file, text="\n  ----- Training K-folds -----")
    # create new Pandas metrics dataframe
    df_metrics_reg, df_metrics_class = pd.DataFrame(), pd.DataFrame()

    # iterate and run each fold number
    for i in range(args.n_folds_run):
        str_current_k = "k" + str(i + 1).zfill(2)
        args_fold.model_name = args.model_name + '/' + str_current_k
        args_fold.out_path = args.out_path + '/' + str_current_k + "/"
        args_fold.log_file = args_fold.out_path + "output.log"
        args_fold.tboard_path = args_fold.out_path + "tboard/"
        args_fold.train_csv_path = args.train_valid_test_path + str_current_k + '.train.csv'
        args_fold.valid_csv_path = args.train_valid_test_path + str_current_k + '.valid.csv'
        args_fold.test_csv_path = args.train_valid_test_path + str_current_k + '.test.csv'

        # exit if the train, valid or text csv files does not exists
        if (Path(args_fold.train_csv_path).is_file()
                and Path(args_fold.valid_csv_path).is_file()
                and Path(args_fold.test_csv_path).is_file()):
            # train fold or skip to the next if the path already exists
            if not Path(args_fold.out_path).is_dir():
                print_file(filename=args.log_file, text='\n  Running experiment fold ' + str(i + 1).zfill(2)
                                                        + ': ' + args_fold.out_path)
                train_fold.main(args_fold)
            else:
                print_file(filename=args.log_file,
                           text='\n  WARN: Output path already exists. Skipping: ' + args_fold.out_path)
        else:
            print_file(filename=args.log_file, text='\n  ERROR: train, validation or test csv files do not exist! Exiting...')
            exit(2)

        # predict and collect metric
        result_metrics_path = args.out_path + '/results/' + str_current_k + '/'
        os.makedirs(result_metrics_path, exist_ok=True)
        args_fold.log_file = result_metrics_path + 'output.log'

        print_file(filename=args.log_file, text='\n  Predicting training set and collecting metrics:')
        print_file(filename=args.log_file, text='  ' + result_metrics_path + 'train_dataset.metrics\n')
        df_tmp_reg, df_tmp_class = predict(args_fold, args_fold.train_csv_path,
                                           metrics_out_filepath=result_metrics_path + 'train_dataset.metrics')
        df_tmp_reg.insert(loc=0, column='fold', value=str_current_k)
        df_tmp_class.insert(loc=0, column='fold', value=str_current_k)
        df_tmp_reg.insert(loc=0, column='partition', value='train_set')
        df_tmp_class.insert(loc=0, column='partition', value='train_set')
        df_metrics_reg = df_metrics_reg.append(df_tmp_reg)
        df_metrics_class = df_metrics_class.append(df_tmp_class)

        print_file(filename=args.log_file, text='\n  Predicting validation set and collecting metrics:')
        print_file(filename=args.log_file, text='  ' + result_metrics_path + 'valid_dataset.metrics\n')
        df_tmp_reg, df_tmp_class = predict(args_fold, args_fold.valid_csv_path,
                                           metrics_out_filepath=result_metrics_path + 'valid_dataset.metrics')
        df_tmp_reg.insert(loc=0, column='fold', value=str_current_k)
        df_tmp_class.insert(loc=0, column='fold', value=str_current_k)
        df_tmp_reg.insert(loc=0, column='partition', value='valid_set')
        df_tmp_class.insert(loc=0, column='partition', value='valid_set')
        df_metrics_reg = df_metrics_reg.append(df_tmp_reg)
        df_metrics_class = df_metrics_class.append(df_tmp_class)
        # operating pointing selection: save validation cutoff_ops to use in test prediction
        if 'cutoff_ops' in df_tmp_class:
            valid_cutoff = df_tmp_class['cutoff_ops'].values
        else:
            valid_cutoff = None

        # print_file(filename=args.log_file, text="\n  Finding operation point selection: validation set")
        # pass

        print_file(filename=args.log_file, text='\n  Predicting test set and collecting metrics:')
        print_file(filename=args.log_file, text='  ' + result_metrics_path + 'test_dataset.metrics\n')
        df_tmp_reg, df_tmp_class = predict(args_fold, args_fold.test_csv_path,
                                           metrics_out_filepath=result_metrics_path + 'test_dataset.metrics',
                                           cutoff_value=valid_cutoff)
        df_tmp_reg.insert(loc=0, column='fold', value=str_current_k)
        df_tmp_class.insert(loc=0, column='fold', value=str_current_k)
        df_tmp_reg.insert(loc=0, column='partition', value='test_set')
        df_tmp_class.insert(loc=0, column='partition', value='test_set')
        df_metrics_reg = df_metrics_reg.append(df_tmp_reg)
        df_metrics_class = df_metrics_class.append(df_tmp_class)

    # clear dataframes' indices
    blank_idx = [''] * len(df_metrics_reg)
    df_metrics_reg.index = blank_idx
    blank_idx = [''] * len(df_metrics_class)
    df_metrics_class.index = blank_idx
    # print resultant dataframes
    print_file(filename=args.log_file, text='\n----- Write metrics to csv files -----')
    print_file(filename=args.log_file, text='  ' + args.out_path + '/results/metrics_regression.csv')
    print_file(filename=args.log_file, text='  ' + args.out_path + '/results/metrics_classification.csv')
    df_metrics_reg.to_csv(args.out_path + '/results/metrics_regression.csv')
    df_metrics_class.to_csv(args.out_path + '/results/metrics_classification.csv')
    print_file(filename=args.log_file, text='\n----- Regression Metrics -----')
    print_file(filename=args.log_file, text=df_metrics_reg.to_string())
    print_file(filename=args.log_file, text='\n----- Classification Metrics -----')
    print_file(filename=args.log_file, text=df_metrics_class.to_string())
    print_file(filename=args.log_file, text="\n----- DONE! -----")


if __name__ == '__main__':
    """
    Usage example:
        docker exec -it sgotf2ctn /usr/bin/python /project/sources/run_experiment.py 
            --project_config_path /project/sources/config/INPD/ 
            --train_valid_test_path /project/data/INPD/train_valid_test/kfold10-gender0cl_tot10-seed88 
            --n_folds_run 2 
            --model_prefix teste123_ 
            --csv_phenotypes /project/data/INPD/phenotypics.csv 
            --seed 77 
            --n_epochs 3 
            --model_arch cnn3d_cole
    """
    shell_args = set_config_run_experiment()
    main(shell_args)

# args: {out_path, tboard_path, log_file, seed, csv_phenotypes, train_csv_path, valid_csv_path, test_csv_path, lr, beta_1, beta_2, model_losses, model_metrics, model_arch, batch_size, n_epochs}
