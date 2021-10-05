import pandas as pd
from utils import *
from tensorflow_dataset import get_assess_dataset, get_dataset_iterator

def main(args):
    full_dataset_csv_path = args.train_valid_test_path + '/full_dataset.csv'
    result_csv = args.out_path + '/voxels_sum.csv'
    examples = pd.read_csv(full_dataset_csv_path).values
    # assign the dataset and iterator
    dataset = get_assess_dataset(args, examples)
    iterator = get_dataset_iterator(dataset, args.batch_size)
    ids_subjects, voxels_sum = [], []
    # begin the predictions
    has_elements = True
    i = 1
    while has_elements:
        try:
            # get one batch
            inputs_it, labels_it = next(iterator)
            # sum the voxels
            voxels_sum.extend(inputs_it['image'].sum(axis=(1, 2, 3, 4)))
            ids_subjects.extend(inputs_it['id'])
            i += 1
        except StopIteration:
            has_elements = False
            pass
    df_voxels_sum = pd.DataFrame(list(zip(ids_subjects, voxels_sum)), columns=['subjectid', 'voxels_sum'])
    df_voxels_sum.to_csv(result_csv, index=False, na_rep='')


if __name__ == '__main__':
    """
    Usage example:
        docker exec -it sgotf2ctn /usr/bin/python /project/sources/calc_voxels_sum.py 
            --project_config_path /project/sources/config/INPD/ 
            --train_valid_test_path /project/data/INPD/train_valid_test/kfold10-gender0cl_tot10-seed88
            --out_path /path/to/generate/csv/
    """
    shell_args = set_config_run_experiment()
    main(shell_args)
