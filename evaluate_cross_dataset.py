from predict import predict
from utils import *
from pathlib import Path
from saliency_maps import generate_saliency_maps, normalize_saliency_maps, save_saliency_brain_nii, map_attention_rois

def main(arguments):

    args = arguments
    # create directory if it does not exists
    os.makedirs(args.result_metrics_path, exist_ok=True)
    # cross dataset metrics filepath
    cross_data_metrics = args.result_metrics_path + 'cross_dataset.metrics'

    if Path(cross_data_metrics).is_file():
        print_file(filename=args.log_file,
                   text='\n  WARN: cross_dataset.metrics already exists. Skipping...')
    else:
        # predict metrics
        print_file(filename=args.log_file, text='\n  Predicting cross dataset and collecting metrics:')
        print_file(filename=args.log_file, text='  ' + cross_data_metrics + '\n')

        df_metrics_reg, df_metrics_class = predict(args, args.full_dataset_csv_path,
                                                   metrics_out_filepath=cross_data_metrics)
        df_metrics_reg.insert(loc=0, column='fold', value='all_data')
        df_metrics_class.insert(loc=0, column='fold', value='all_data')
        df_metrics_reg.insert(loc=0, column='partition', value='cross_dataset')
        df_metrics_class.insert(loc=0, column='partition', value='cross_dataset')
        # clear dataframes' indices
        blank_idx = [''] * len(df_metrics_reg)
        df_metrics_reg.index = blank_idx
        blank_idx = [''] * len(df_metrics_class)
        df_metrics_class.index = blank_idx
        # print resultant dataframes
        print_file(filename=args.log_file, text='\n----- Write metrics to csv files -----')
        print_file(filename=args.log_file, text='  ' + args.result_metrics_path + 'metrics_regression.csv')
        print_file(filename=args.log_file, text='  ' + args.result_metrics_path + 'metrics_classification.csv')
        df_metrics_reg.to_csv(args.result_metrics_path + 'metrics_regression.csv')
        df_metrics_class.to_csv(args.result_metrics_path + 'metrics_classification.csv')

    if Path(args.result_metrics_path + 'saliency_mean.npy').is_file():
        print_file(filename=args.log_file,
                   text='\n  WARN: saliency_mean.npy from cross daraset already exists. Skipping...')
    else:
        # saliency maps list
        saliency_list, brain_list = [], []
        print_file(filename=args.log_file, text='\n  Generating cross dataset salience and brain maps (SmoothGrad):')
        saliency_mean, brain_mean = generate_saliency_maps(args, examples_to_map=args.full_dataset_csv_path)
        saliency_list.append(saliency_mean)
        brain_list.append(brain_mean)
        print_file(filename=args.log_file, text='  ' + args.result_metrics_path + 'saliency_mean.npy')
        np.save(args.result_metrics_path + 'saliency_mean.npy', saliency_mean)
        print_file(filename=args.log_file, text='  ' + args.result_metrics_path + 'brain_mean.npy')
        np.save(args.result_metrics_path + 'brain_mean.npy', brain_mean)
        print_file(filename=args.log_file, text='  ' + args.result_metrics_path + 'saliency_mean.nii')
        print_file(filename=args.log_file, text='  ' + args.result_metrics_path + 'brain_mean_gm.nii')
        print_file(filename=args.log_file, text='  ' + args.result_metrics_path + 'brain_mean_wm.nii')
        save_saliency_brain_nii(args.result_metrics_path, saliency_mean, brain_mean)

    print_file(filename=args.log_file, text="\n----- DONE! -----")


if __name__ == '__main__':
    """
    Usage example:
        docker exec -it sgotf2ctn /usr/bin/python /project/sources/evaluate_cross_dataset.py 
            --saved_model_path /project/output/INPD/teste123_kfold10-gender0cl_tot10-seed88/k01/ 
            --full_dataset_csv_path /project/data/INPD/train_valid_test/kfold10-gender0cl_tot10-seed99/full_dataset.csv 
            --config_model_path /project/sources/config/INPD/ 
            --config_dataset_path /project/sources/config/INPD/
    """
    shell_args = set_config_evaluate_cross_dataset()
    main(shell_args)
