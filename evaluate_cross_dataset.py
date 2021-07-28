from predict import predict
from utils import *

def main(arguments):

    args = arguments
    # create directory if it does not exists
    os.makedirs(args.result_metrics_path, exist_ok=True)
    # predict metrics
    print_file(filename=args.log_file, text='\n  Predicting cross dataset and collecting metrics:')
    print_file(filename=args.log_file, text='  ' + args.result_metrics_path + 'cross_dataset.metrics\n')
    predict(args, args.full_dataset_csv_path, metrics_out_filepath=args.result_metrics_path + 'cross_dataset.metrics')
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
