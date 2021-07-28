from sklearn.model_selection import StratifiedKFold
import pandas as pd
from utils import *


def generate_stratified_kfold(args, out_log):
    """
    Generate a custom stratified kfold validation containing train, validation and test for each k round.

    :param args: dictionary of configuration parameters
    :param out_log: filepath to write log output
    """

    # load phenotypes in dataframe
    df = pd.read_csv(args.csv_phenotypes)
    # read number of partitions k
    k = args.kfold

    print_file(filename=out_log, text='\n----- Generate the Kfold partitions -----')
    print_file(filename=out_log, text='  Number of folds = ' + str(k))
    print_file(filename=out_log, text='  Stratification schema = ' + args.stratify_config)

    # prepare features transforming continuous variables in categories
    stratify_columns = []  # final columns to stratify samples
    num_continuous = 0  # counter of continuous features
    for column, qtl_num in eval(args.stratify_config).items():
        if qtl_num == 0:
            # just append categorical columns
            stratify_columns.append(column)
        else:
            # increment and generate category from continuous features
            num_continuous += 1
            # create generated_feature and assign it a default value
            generated_feature = 'generated_cat_' + str(num_continuous)
            df[generated_feature] = -777 #pd.Series(dtype='int64')
            # number of quantiles to split
            n_quantiles = qtl_num
            # create percentiles based on the number of quantiles to split
            percentiles_split = np.linspace(0, 1, n_quantiles + 1)

            for keys, idx in df.groupby(stratify_columns).groups.items():
                index = idx.values.copy()
                # create the quantiles
                quantiles = df.loc[index, column].quantile(percentiles_split).unique()
                # create the split conditions based on each quantile interval
                conditions = []
                for i in range(len(quantiles) - 1):
                    conditions.append(df.loc[index, column].between(quantiles[i], quantiles[i + 1]))
                # create the categorical values based on the length of conditions
                cat_values = np.arange(len(conditions))
                # assign the generated categorical values to a generated feature in df
                df.loc[index, generated_feature] = np.select(conditions, cat_values, default=-999)
            # append the previous generated feature to stratify_columns
            stratify_columns.append(generated_feature)
    # stratify_columns = ["state", "gender", "generated_cat_1"]  # ["state", "gender", "dcany"]

    # store the location of each tensorflow record in a column named filename
    df['filename'] = df['FNAME_TFREC'].values

    # stratify_cats will store the combined generated category which will be used for final stratification
    df['stratify_cats'] = pd.Series(dtype='int64')
    # stratify_keys will store the concatenated values of the categories used for stratification
    df['stratify_keys'] = pd.Series(dtype='object')
    for enum, (keys, idx) in enumerate(df.groupby(stratify_columns).groups.items()):
        index = idx.values.copy()
        df.loc[index, 'stratify_cats'] = enum
        df.loc[index, 'stratify_keys'] = str(keys)

    print_file(filename=out_log, text='\n----- Stratification totals -----')
    df_result = df.groupby(['stratify_keys'])['stratify_keys'].count().reset_index(name="total")
    df_result.columns = ["(" + ", ".join(stratify_columns) + ")", "total"]
    print_file(filename=out_log, text=df_result.to_string(index=False))

    print_file(filename=out_log, text='\n----- Generated csv files -----')

    # generate full dataset csv (for cross-dataset validation)
    csv_filename_full_dataset = args.out_path_splits + "full_dataset.csv"
    df.to_csv(csv_filename_full_dataset, index=False, na_rep='', columns=['filename'])

    # create stratification
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
    skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
    # classes stratification
    for m, (train_index, valid_test_index) in enumerate(skf.split(df, df['stratify_cats'])):
        # create train, validation and test csv filenames for the current folder
        csv_filename_train = args.out_path_splits + "k" + str(m + 1).zfill(2) + ".train.csv"
        csv_filename_valid = args.out_path_splits + "k" + str(m + 1).zfill(2) + ".valid.csv"
        csv_filename_test = args.out_path_splits + "k" + str(m + 1).zfill(2) + ".test.csv"

        # obtain train and valid_test partitions
        df_train, df_valid_test = df.iloc[train_index], df.iloc[valid_test_index]
        # write train filenames to csv
        print_file(filename=out_log, text='  ' + csv_filename_train)
        df_train.to_csv(csv_filename_train, index=False, na_rep='', columns=['filename'])

        # split valid_test in validation and test partitions
        for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['stratify_cats'])):
            if n == 1:
                df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
                # write validation filenames to csv
                print_file(filename=out_log, text='  ' + csv_filename_valid)
                df_valid.to_csv(csv_filename_valid, index=False, na_rep='', columns=['filename'])
                # write test filenames to csv
                print_file(filename=out_log, text='  ' + csv_filename_test)
                df_test.to_csv(csv_filename_test, index=False, na_rep='', columns=['filename'])


if __name__ == '__main__':
    """ 
    Usage example: 
        docker exec -it sgotf2ctn /usr/bin/python /project/sources/generate_data_splits.py 
            --project_config_path "/project/sources/config/INPD/" --kfold 10 --seed 88 
            --stratify_config "{'gender':0, 'cl_tot': 10}"
    """

    # set configuration parameters
    ARGS = set_config_generate_splits()
    # assign output_log filename
    output_log = ARGS.out_path_splits + 'output.log'
    # remove output_log if it already exists
    os.remove(output_log) if os.path.exists(output_log) else None

    # create output directory to store csv output splits
    try:
        print(ARGS.out_path_splits)
        os.makedirs(ARGS.out_path_splits, exist_ok=True)
    except OSError:
        print("\nCreation of directories failed.")
    else:
        print_file(filename=output_log, text="\nSuccessfully created directories.")

    # print execution parameters
    print_file(filename=output_log, text="\n----- Execution parameters -----")
    for key, val in vars(ARGS).items():
        print_file(filename=output_log, text=f"  {key}: {val}")

    # generate data split partitions
    generate_stratified_kfold(ARGS, output_log)

    print_file(filename=output_log, text="\n----- DONE! -----")


# def generate_kfold_v6(dframe, k, chosen_k):
#     """
#     Generate a custom kfold validation containing train_pos, train_neg, validation and test for each round.
#
#     :param dframe: pandas dataframe with phenotype information
#     :param k: number of kfold partitions
#     :param chosen_k: chosen k round to return the partitions data
#     :return: dictionary containing [train_pos, train_neg, validation, test] partitions from chosen K round
#     """
#
#     # stratify samples and shuffle image files
#     # dframe['filename'] = ARGS.image_location_prefix + dframe['FNAME_TFREC'].values  # store filename
#     dframe['filename'] = dframe['FNAME_TFREC'].values  # store filename
#
#     n_bins = 10
#
#     percentiles_split = np.linspace(0, 1, n_bins+1)
#     quantiles = dframe['cl_tot'].quantile(percentiles_split).unique()
#
#     conditions = []
#     for i in range(len(quantiles) - 1):
#         if i < (len(quantiles) - 2):
#             conditions.append(dframe['cl_tot'].between(quantiles[i], quantiles[i+1] - 0.001))
#         else:
#             conditions.append(dframe['cl_tot'].between(quantiles[i], quantiles[i+1]))
#     cat_values = np.arange(len(conditions))
#     dframe['generated_cat_1'] = np.select(conditions, cat_values, default=-999)
#
#     stratify_columns = ["state", "gender", "generated_cat_1"]  # ["state", "gender", "dcany"]
#
#     df['stratify_cats'] = pd.Series(dtype='int64')
#     df['stratify_keys'] = pd.Series(dtype='object')
#     for enum, (key, idx) in enumerate(df.groupby(stratify_columns).groups.items()):
#         index = idx.values.copy()
#         df.loc[index, 'stratify_cats'] = enum
#         df.loc[index, 'stratify_keys'] = str(key)
#
#     print_file(filename=ARGS.log_file, text='----- Stratification totals -----')
#     df_result = df.groupby(['stratify_keys'])['stratify_keys'].count().reset_index(name="total")
#     df_result.columns = ["(" + ", ".join(stratify_columns) + ")", "total"]
#     print_file(filename=ARGS.log_file, text=df_result.to_string(index=False))
#
#     fold_file_name = ARGS.out_path + "kfold" + str(ARGS.chosen_k) + "_examples.txt"
#     print_file(filename=ARGS.log_file, text='\n----- Generate and return chosen Kfold partitions -----')
#     print_file(filename=ARGS.log_file, text='  Number of folds = ' + str(k))
#     print_file(filename=ARGS.log_file, text='  Chosen K partitions = ' + str(chosen_k))
#     # each item of theses lists is a partition of a given fold number
#     train_list, valid_list, test_list = [], [], []
#
#     # create stratification
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=ARGS.seed)
#     skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=ARGS.seed)
#
#     # classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(dframe, dframe['stratify_cats'])):
#         df_train, df_valid_test = dframe.iloc[train_index], dframe.iloc[valid_test_index]
#         train_list.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['stratify_cats'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list.append(df_valid['filename'])
#                 test_list.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\n---Partitions of K = ' + str(m+1), fold_file_name)
#             print_file('\ntrain_list[' + str(m+1) + ']\n' + '\n'.join(train_list[m]), fold_file_name)
#             print_file('\nvalid_list[' + str(m+1) + ']\n' + '\n'.join(valid_list[m]), fold_file_name)
#             print_file('\ntest_list[' + str(m+1) + ']\n' + '\n'.join(test_list[m]), fold_file_name)
#
#     # assign the chosen K partitions to resulting variables
#     train_fns = train_list[chosen_k - 1].tolist()
#     validation_fns = valid_list[chosen_k - 1].tolist()
#     test_fns = test_list[chosen_k - 1].tolist()
#
#     # shuffle the results
#     random.shuffle(train_fns)
#     random.shuffle(validation_fns)
#     random.shuffle(test_fns)
#
#     # store results in a dict
#     result_dict = dict()
#     result_dict['train_fns'] = train_fns
#     result_dict['validation_fns'] = validation_fns
#     result_dict['test_fns'] = test_fns
#
#     return result_dict

# def generate_kfold_v0(df, k, chosen_k):
#     """
#     Generate a custom kfold validation containing train_pos, train_neg, validation and test for each round.
# 
#     :param df: pandas dataframe with phenotype information
#     :param k: number of kfold partitions
#     :param chosen_k: chosen k round to return the partitions data
#     :return: dictionary containing [train_pos, train_neg, validation, test] partitions from chosen K round
#     """
#     # stratify samples and shuffle image files
#     # df['filename'] = args.image_location_prefix + df['FNAME_TFREC'].values #store filename
#     df['filename'] = df['FNAME_TFREC'].values  # store filename
# 
#     df_pos = df[df['gender'] == 1].copy()  # male
#     df_neg = df[df['gender'] == 2].copy()  # female
# 
#     fold_file_name = args.out_path + "kfold" + str(args.chosen_k) + "_examples.txt"
#     print_file(filename=args.log_file, text='\n----- Generate and return chosen Kfold partitions -----')
#     print_file(filename=args.log_file, text='  Number of folds = ' + str(k))
#     print_file(filename=args.log_file, text='  Chosen K partitions = ' + str(chosen_k))
#     # each item of theses lists is a partition of a given fold number
#     train_list_pos, train_list_neg = [], []
#     valid_list_pos, valid_list_neg = [], []
#     test_list_pos, test_list_neg = [], []
#     # create stratification
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
#     skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
# 
#     # positive classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_pos, df_pos['dcany'])):
#         df_train, df_valid_test = df_pos.iloc[train_index], df_pos.iloc[valid_test_index]
#         train_list_pos.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         # print(df_valid_test.groupby('cl_tot_bin').agg(['count'])['filename'])
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['dcany'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_pos.append(df_valid['filename'])
#                 test_list_pos.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\n---Positive partitions of K = ' + str(m+1), fold_file_name)
#             print_file('\ntrain_list_pos[' + str(m+1) + ']\n' + '\n'.join(train_list_pos[m]), fold_file_name)
#             print_file('\nvalid_list_pos[' + str(m+1) + ']\n' + '\n'.join(valid_list_pos[m]), fold_file_name)
#             print_file('\ntest_list_pos[' + str(m+1) + ']\n' + '\n'.join(test_list_pos[m]), fold_file_name)
# 
#     # negative classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_neg, df_neg['dcany'])):
#         df_train, df_valid_test = df_neg.iloc[train_index], df_neg.iloc[valid_test_index]
#         train_list_neg.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['dcany'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_neg.append(df_valid['filename'])
#                 test_list_neg.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\ntrain_list_neg[' + str(m+1) + ']\n' + '\n'.join(train_list_neg[m]), fold_file_name)
#             print_file('\nvalid_list_neg[' + str(m+1) + ']\n' + '\n'.join(valid_list_neg[m]), fold_file_name)
#             print_file('\ntest_list_neg[' + str(m+1) + ']\n' + '\n'.join(test_list_neg[m]), fold_file_name)
# 
#     # assign the chosen K partitions to resulting variables
#     train_fns_pos = train_list_pos[chosen_k - 1].tolist()
#     train_fns_neg = train_list_neg[chosen_k - 1].tolist()
#     validation_fns_pos = valid_list_pos[chosen_k - 1].tolist()
#     validation_fns_neg = valid_list_neg[chosen_k - 1].tolist()
#     validation_fns = valid_list_pos[chosen_k - 1].tolist() + valid_list_neg[chosen_k - 1].tolist()
#     test_fns_pos = test_list_pos[chosen_k - 1].tolist()
#     test_fns_neg = test_list_neg[chosen_k - 1].tolist()
#     test_fns = test_list_pos[chosen_k - 1].tolist() + test_list_neg[chosen_k - 1].tolist()
# 
#     # shuffle the results
#     random.shuffle(train_fns_pos)
#     random.shuffle(train_fns_neg)
#     random.shuffle(validation_fns)
#     random.shuffle(test_fns)
# 
#     # store results in a dict
#     result_dict = dict()
#     result_dict['train_fns_pos'] = train_fns_pos
#     result_dict['train_fns_neg'] = train_fns_neg
#     result_dict['validation_fns_neg'] = validation_fns_neg
#     result_dict['validation_fns_pos'] = validation_fns_pos
#     result_dict['validation_fns'] = validation_fns
#     result_dict['test_fns_neg'] = test_fns_neg
#     result_dict['test_fns_pos'] = test_fns_pos
#     result_dict['test_fns'] = test_fns
# 
#     return result_dict
# 
# def generate_kfold_v1(df, k, chosen_k):
#     """
#     Generate a custom kfold validation containing train_pos, train_neg, validation and test for each round.
# 
#     :param df: pandas dataframe with phenotype information
#     :param k: number of kfold partitions
#     :param chosen_k: chosen k round to return the partitions data
#     :return: dictionary containing [train_pos, train_neg, validation, test] partitions from chosen K round
#     """
#     # stratify samples and shuffle image files
#     # df['filename'] = args.image_location_prefix + df['FNAME_TFREC'].values #store filename
#     df['filename'] = df['FNAME_TFREC'].values  # store filename
# 
#     bin_values = [1, 2, 3, 4, 5, 6, 6, 6, 6]
# 
#     df_pos = df[df['gender'] == 1].copy() #male
#     _, bins1 = np.histogram(df_pos['cl_tot'], bins=9)
#     conditions1 = [
#         df_pos['cl_tot'].between(bins1[0], bins1[1] - 1),
#         df_pos['cl_tot'].between(bins1[1], bins1[2] - 1),
#         df_pos['cl_tot'].between(bins1[2], bins1[3] - 1),
#         df_pos['cl_tot'].between(bins1[3], bins1[4] - 1),
#         df_pos['cl_tot'].between(bins1[4], bins1[5] - 1),
#         df_pos['cl_tot'].between(bins1[5], bins1[6] - 1),
#         df_pos['cl_tot'].between(bins1[6], bins1[7] - 1),
#         df_pos['cl_tot'].between(bins1[7], bins1[8] - 1),
#         df_pos['cl_tot'].between(bins1[8], bins1[9])]
#     df_pos['cl_tot_bin'] = np.select(conditions1, bin_values, default=0)
# 
#     df_neg = df[df['gender'] == 2].copy()  #female
#     _, bins2 = np.histogram(df_neg['cl_tot'], bins=9)
#     conditions2 = [
#         df_neg['cl_tot'].between(bins2[0], bins2[1] - 1),
#         df_neg['cl_tot'].between(bins2[1], bins2[2] - 1),
#         df_neg['cl_tot'].between(bins2[2], bins2[3] - 1),
#         df_neg['cl_tot'].between(bins2[3], bins2[4] - 1),
#         df_neg['cl_tot'].between(bins2[4], bins2[5] - 1),
#         df_neg['cl_tot'].between(bins2[5], bins2[6] - 1),
#         df_neg['cl_tot'].between(bins2[6], bins2[7] - 1),
#         df_neg['cl_tot'].between(bins2[7], bins2[8] - 1),
#         df_neg['cl_tot'].between(bins2[8], bins2[9])]
#     df_neg['cl_tot_bin'] = np.select(conditions2, bin_values, default=0)
# 
#     fold_file_name = args.out_path + "kfold" + str(args.chosen_k) + "_examples.txt"
#     print_file(filename=args.log_file, text='\n----- Generate and return chosen Kfold partitions -----')
#     print_file(filename=args.log_file, text='  Number of folds = ' + str(k))
#     print_file(filename=args.log_file, text='  Chosen K partitions = ' + str(chosen_k))
#     # each item of theses lists is a partition of a given fold number
#     train_list_pos, train_list_neg = [], []
#     valid_list_pos, valid_list_neg = [], []
#     test_list_pos, test_list_neg = [], []
#     # create stratification
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
#     skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
# 
#     # positive classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_pos, df_pos['cl_tot_bin'])):
#         df_train, df_valid_test = df_pos.iloc[train_index], df_pos.iloc[valid_test_index]
#         train_list_pos.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         # print(df_valid_test.groupby('cl_tot_bin').agg(['count'])['filename'])
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_tot_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_pos.append(df_valid['filename'])
#                 test_list_pos.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\n---Positive partitions of K = ' + str(m+1), fold_file_name)
#             print_file('\ntrain_list_pos[' + str(m+1) + ']\n' + '\n'.join(train_list_pos[m]), fold_file_name)
#             print_file('\nvalid_list_pos[' + str(m+1) + ']\n' + '\n'.join(valid_list_pos[m]), fold_file_name)
#             print_file('\ntest_list_pos[' + str(m+1) + ']\n' + '\n'.join(test_list_pos[m]), fold_file_name)
# 
#     # negative classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_neg, df_neg['cl_tot_bin'])):
#         df_train, df_valid_test = df_neg.iloc[train_index], df_neg.iloc[valid_test_index]
#         train_list_neg.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_tot_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_neg.append(df_valid['filename'])
#                 test_list_neg.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\ntrain_list_neg[' + str(m+1) + ']\n' + '\n'.join(train_list_neg[m]), fold_file_name)
#             print_file('\nvalid_list_neg[' + str(m+1) + ']\n' + '\n'.join(valid_list_neg[m]), fold_file_name)
#             print_file('\ntest_list_neg[' + str(m+1) + ']\n' + '\n'.join(test_list_neg[m]), fold_file_name)
# 
#     # assign the chosen K partitions to resulting variables
#     train_fns_pos = train_list_pos[chosen_k - 1].tolist()
#     train_fns_neg = train_list_neg[chosen_k - 1].tolist()
#     validation_fns_pos = valid_list_pos[chosen_k - 1].tolist()
#     validation_fns_neg = valid_list_neg[chosen_k - 1].tolist()
#     validation_fns = valid_list_pos[chosen_k - 1].tolist() + valid_list_neg[chosen_k - 1].tolist()
#     test_fns_pos = test_list_pos[chosen_k - 1].tolist()
#     test_fns_neg = test_list_neg[chosen_k - 1].tolist()
#     test_fns = test_list_pos[chosen_k - 1].tolist() + test_list_neg[chosen_k - 1].tolist()
# 
#     # shuffle the results
#     random.shuffle(train_fns_pos)
#     random.shuffle(train_fns_neg)
#     random.shuffle(validation_fns)
#     random.shuffle(test_fns)
# 
#     # store results in a dict
#     result_dict = dict()
#     result_dict['train_fns_pos'] = train_fns_pos
#     result_dict['train_fns_neg'] = train_fns_neg
#     result_dict['validation_fns_neg'] = validation_fns_neg
#     result_dict['validation_fns_pos'] = validation_fns_pos
#     result_dict['validation_fns'] = validation_fns
#     result_dict['test_fns_neg'] = test_fns_neg
#     result_dict['test_fns_pos'] = test_fns_pos
#     result_dict['test_fns'] = test_fns
# 
#     return result_dict
# 
# def generate_kfold_v2(df, k, chosen_k):
#     """
#     Generate a custom kfold validation containing train_pos, train_neg, validation and test for each round.
# 
#     :param df: pandas dataframe with phenotype information
#     :param k: number of kfold partitions
#     :param chosen_k: chosen k round to return the partitions data
#     :return: dictionary containing [train_pos, train_neg, validation, test] partitions from chosen K round
#     """
# 
#     # stratify samples and shuffle image files
#     # df['filename'] = args.image_location_prefix + df['FNAME_TFREC'].values #store filename
#     df['filename'] = df['FNAME_TFREC'].values  # store filename
# 
#     bin_values = [1, 2, 3, 4, 5, 6, 7, 8, 8, 8]
# 
#     df_pos = df
#     _, bins1 = np.histogram(df_pos['cl_tot'], bins=10)
#     conditions1 = [
#         df_pos['cl_tot'].between(bins1[0], bins1[1] - 1),
#         df_pos['cl_tot'].between(bins1[1], bins1[2] - 1),
#         df_pos['cl_tot'].between(bins1[2], bins1[3] - 1),
#         df_pos['cl_tot'].between(bins1[3], bins1[4] - 1),
#         df_pos['cl_tot'].between(bins1[4], bins1[5] - 1),
#         df_pos['cl_tot'].between(bins1[5], bins1[6] - 1),
#         df_pos['cl_tot'].between(bins1[6], bins1[7] - 1),
#         df_pos['cl_tot'].between(bins1[7], bins1[8] - 1),
#         df_pos['cl_tot'].between(bins1[8], bins1[9] - 1),
#         df_pos['cl_tot'].between(bins1[9], bins1[10])
#     ]
#     df_pos['cl_tot_bin'] = np.select(conditions1, bin_values, default=0)
# 
#     fold_file_name = args.out_path + "kfold" + str(args.chosen_k) + "_examples.txt"
#     print_file(filename=args.log_file, text='\n----- Generate and return chosen Kfold partitions -----')
#     print_file(filename=args.log_file, text='  Number of folds = ' + str(k))
#     print_file(filename=args.log_file, text='  Chosen K partitions = ' + str(chosen_k))
#     # each item of theses lists is a partition of a given fold number
#     train_list_pos, train_list_neg = [], []
#     valid_list_pos, valid_list_neg = [], []
#     test_list_pos, test_list_neg = [], []
#     # create stratification
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
#     skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
# 
#     # positive classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_pos, df_pos['cl_tot_bin'])):
#         df_train, df_valid_test = df_pos.iloc[train_index], df_pos.iloc[valid_test_index]
#         train_list_pos.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         # print(df_valid_test.groupby('cl_tot_bin').agg(['count'])['filename'])
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_tot_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_pos.append(df_valid['filename'])
#                 test_list_pos.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\n---Positive partitions of K = ' + str(m+1), fold_file_name)
#             print_file('\ntrain_list_pos[' + str(m+1) + ']\n' + '\n'.join(train_list_pos[m]), fold_file_name)
#             print_file('\nvalid_list_pos[' + str(m+1) + ']\n' + '\n'.join(valid_list_pos[m]), fold_file_name)
#             print_file('\ntest_list_pos[' + str(m+1) + ']\n' + '\n'.join(test_list_pos[m]), fold_file_name)
# 
# 
#     # assign the chosen K partitions to resulting variables
#     train_fns_pos = train_list_pos[chosen_k - 1].tolist()
#     train_fns_neg = []
#     validation_fns_pos = valid_list_pos[chosen_k - 1].tolist()
#     validation_fns_neg = []
#     validation_fns = validation_fns_pos + validation_fns_neg
#     test_fns_pos = test_list_pos[chosen_k - 1].tolist()
#     test_fns_neg = []
#     test_fns = test_fns_pos + test_fns_neg
# 
#     # shuffle the results
#     random.shuffle(train_fns_pos)
#     random.shuffle(train_fns_neg)
#     random.shuffle(validation_fns)
#     random.shuffle(test_fns)
# 
#     # store results in a dict
#     result_dict = dict()
#     result_dict['train_fns_pos'] = train_fns_pos
#     result_dict['train_fns_neg'] = train_fns_neg
#     result_dict['validation_fns_neg'] = validation_fns_neg
#     result_dict['validation_fns_pos'] = validation_fns_pos
#     result_dict['validation_fns'] = validation_fns
#     result_dict['test_fns_neg'] = test_fns_neg
#     result_dict['test_fns_pos'] = test_fns_pos
#     result_dict['test_fns'] = test_fns
# 
#     return result_dict
# 
# def generate_kfold_v3(df, k, chosen_k):
#     """
#     Generate a custom kfold validation containing train_pos, train_neg, validation and test for each round.
# 
#     :param df: pandas dataframe with phenotype information
#     :param k: number of kfold partitions
#     :param chosen_k: chosen k round to return the partitions data
#     :return: dictionary containing [train_pos, train_neg, validation, test] partitions from chosen K round
#     """
# 
#     # stratify samples and shuffle image files
#     # df['filename'] = args.image_location_prefix + df['FNAME_TFREC'].values #store filename
#     df['filename'] = df['FNAME_TFREC'].values  # store filename
# 
#     bin_values = [1, 2, 3, 4]
# 
#     df_pos = df[df['state'] == 1].copy() #state 1
#     quantiles = df_pos['age'].quantile([0, 0.25, 0.5, 0.75, 1]).values
# 
#     conditions1 = [
#         df_pos['age'].between(quantiles[0], quantiles[1] - 0.001),
#         df_pos['age'].between(quantiles[1], quantiles[2] - 0.001),
#         df_pos['age'].between(quantiles[2], quantiles[3] - 0.001),
#         df_pos['age'].between(quantiles[3], quantiles[4])]
#     df_pos['cl_age_bin'] = np.select(conditions1, bin_values, default=0)
# 
#     df_neg = df[df['state'] == 2].copy()  #state 2
#     quantiles = df_neg['age'].quantile([0, 0.25, 0.5, 0.75, 1]).values
# 
#     conditions2 = [
#         df_neg['age'].between(quantiles[0], quantiles[1] - 0.001),
#         df_neg['age'].between(quantiles[1], quantiles[2] - 0.001),
#         df_neg['age'].between(quantiles[2], quantiles[3] - 0.001),
#         df_neg['age'].between(quantiles[3], quantiles[4])]
#     df_neg['cl_age_bin'] = np.select(conditions2, bin_values, default=0)
# 
#     fold_file_name = args.out_path + "kfold" + str(args.chosen_k) + "_examples.txt"
#     print_file(filename=args.log_file, text='\n----- Generate and return chosen Kfold partitions -----')
#     print_file(filename=args.log_file, text='  Number of folds = ' + str(k))
#     print_file(filename=args.log_file, text='  Chosen K partitions = ' + str(chosen_k))
#     # each item of theses lists is a partition of a given fold number
#     train_list_pos, train_list_neg = [], []
#     valid_list_pos, valid_list_neg = [], []
#     test_list_pos, test_list_neg = [], []
#     # create stratification
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
#     skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
# 
#     # positive classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_pos, df_pos['cl_age_bin'])):
#         df_train, df_valid_test = df_pos.iloc[train_index], df_pos.iloc[valid_test_index]
#         train_list_pos.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         # print(df_valid_test.groupby('cl_tot_bin').agg(['count'])['filename'])
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_age_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_pos.append(df_valid['filename'])
#                 test_list_pos.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\n---Positive partitions of K = ' + str(m+1), fold_file_name)
#             print_file('\ntrain_list_pos[' + str(m+1) + ']\n' + '\n'.join(train_list_pos[m]), fold_file_name)
#             print_file('\nvalid_list_pos[' + str(m+1) + ']\n' + '\n'.join(valid_list_pos[m]), fold_file_name)
#             print_file('\ntest_list_pos[' + str(m+1) + ']\n' + '\n'.join(test_list_pos[m]), fold_file_name)
# 
#     # negative classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_neg, df_neg['cl_age_bin'])):
#         df_train, df_valid_test = df_neg.iloc[train_index], df_neg.iloc[valid_test_index]
#         train_list_neg.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_age_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_neg.append(df_valid['filename'])
#                 test_list_neg.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\ntrain_list_neg[' + str(m+1) + ']\n' + '\n'.join(train_list_neg[m]), fold_file_name)
#             print_file('\nvalid_list_neg[' + str(m+1) + ']\n' + '\n'.join(valid_list_neg[m]), fold_file_name)
#             print_file('\ntest_list_neg[' + str(m+1) + ']\n' + '\n'.join(test_list_neg[m]), fold_file_name)
# 
#     # assign the chosen K partitions to resulting variables
#     train_fns_pos = train_list_pos[chosen_k - 1].tolist()
#     train_fns_neg = train_list_neg[chosen_k - 1].tolist()
#     validation_fns_pos = valid_list_pos[chosen_k - 1].tolist()
#     validation_fns_neg = valid_list_neg[chosen_k - 1].tolist()
#     validation_fns = valid_list_pos[chosen_k - 1].tolist() + valid_list_neg[chosen_k - 1].tolist()
#     test_fns_pos = test_list_pos[chosen_k - 1].tolist()
#     test_fns_neg = test_list_neg[chosen_k - 1].tolist()
#     test_fns = test_list_pos[chosen_k - 1].tolist() + test_list_neg[chosen_k - 1].tolist()
# 
#     # shuffle the results
#     random.shuffle(train_fns_pos)
#     random.shuffle(train_fns_neg)
#     random.shuffle(validation_fns)
#     random.shuffle(test_fns)
# 
#     # store results in a dict
#     result_dict = dict()
#     result_dict['train_fns_pos'] = train_fns_pos
#     result_dict['train_fns_neg'] = train_fns_neg
#     result_dict['validation_fns_neg'] = validation_fns_neg
#     result_dict['validation_fns_pos'] = validation_fns_pos
#     result_dict['validation_fns'] = validation_fns
#     result_dict['test_fns_neg'] = test_fns_neg
#     result_dict['test_fns_pos'] = test_fns_pos
#     result_dict['test_fns'] = test_fns
# 
#     return result_dict
# 
# def generate_kfold_v4(df, k, chosen_k):
#     """
#     Generate a custom kfold validation containing train_pos, train_neg, validation and test for each round.
# 
#     :param df: pandas dataframe with phenotype information
#     :param k: number of kfold partitions
#     :param chosen_k: chosen k round to return the partitions data
#     :return: dictionary containing [train_pos, train_neg, validation, test] partitions from chosen K round
#     """
# 
#     # stratify samples and shuffle image files
#     # df['filename'] = args.image_location_prefix + df['FNAME_TFREC'].values  # store filename
#     df['filename'] = df['FNAME_TFREC'].values  # store filename
# 
#     n_bins = 15
# 
#     bin_values = np.arange(n_bins)+1
#     percentiles_split = np.linspace(0, 1, n_bins+1)
# 
#     df_pos = df[df['state'] == 1].copy()  # state 1
#     df_neg = df[df['state'] == 2].copy()  # state 2
#     quantiles_pos = df_pos['cl_tot'].quantile(percentiles_split).values
#     quantiles_neg = df_neg['cl_tot'].quantile(percentiles_split).values
# 
#     conditions_pos, conditions_neg = [], []
#     for i in range(n_bins):
#         if i < (n_bins - 2):
#             conditions_pos.append(df_pos['cl_tot'].between(quantiles_pos[i], quantiles_pos[i+1] - 0.001))
#             conditions_neg.append(df_neg['cl_tot'].between(quantiles_neg[i], quantiles_neg[i+1] - 0.001))
#         else:
#             conditions_pos.append(df_pos['cl_tot'].between(quantiles_pos[i], quantiles_pos[i+1]))
#             conditions_neg.append(df_neg['cl_tot'].between(quantiles_neg[i], quantiles_neg[i+1]))
# 
#     df_pos['cl_tot_bin'] = np.select(conditions_pos, bin_values, default=0)
#     df_neg['cl_tot_bin'] = np.select(conditions_neg, bin_values, default=0)
# 
#     fold_file_name = args.out_path + "kfold" + str(args.chosen_k) + "_examples.txt"
#     print_file(filename=args.log_file, text='\n----- Generate and return chosen Kfold partitions -----')
#     print_file(filename=args.log_file, text='  Number of folds = ' + str(k))
#     print_file(filename=args.log_file, text='  Chosen K partitions = ' + str(chosen_k))
#     # each item of theses lists is a partition of a given fold number
#     train_list_pos, train_list_neg = [], []
#     valid_list_pos, valid_list_neg = [], []
#     test_list_pos, test_list_neg = [], []
#     # create stratification
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
#     skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
# 
#     # positive classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_pos, df_pos['cl_tot_bin'])):
#         df_train, df_valid_test = df_pos.iloc[train_index], df_pos.iloc[valid_test_index]
#         train_list_pos.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         # print(df_valid_test.groupby('cl_tot_bin').agg(['count'])['filename'])
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_tot_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_pos.append(df_valid['filename'])
#                 test_list_pos.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\n---Positive partitions of K = ' + str(m+1), fold_file_name)
#             print_file('\ntrain_list_pos[' + str(m+1) + ']\n' + '\n'.join(train_list_pos[m]), fold_file_name)
#             print_file('\nvalid_list_pos[' + str(m+1) + ']\n' + '\n'.join(valid_list_pos[m]), fold_file_name)
#             print_file('\ntest_list_pos[' + str(m+1) + ']\n' + '\n'.join(test_list_pos[m]), fold_file_name)
# 
#     # negative classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_neg, df_neg['cl_tot_bin'])):
#         df_train, df_valid_test = df_neg.iloc[train_index], df_neg.iloc[valid_test_index]
#         train_list_neg.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_tot_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_neg.append(df_valid['filename'])
#                 test_list_neg.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\ntrain_list_neg[' + str(m+1) + ']\n' + '\n'.join(train_list_neg[m]), fold_file_name)
#             print_file('\nvalid_list_neg[' + str(m+1) + ']\n' + '\n'.join(valid_list_neg[m]), fold_file_name)
#             print_file('\ntest_list_neg[' + str(m+1) + ']\n' + '\n'.join(test_list_neg[m]), fold_file_name)
# 
#     # assign the chosen K partitions to resulting variables
#     train_fns_pos = train_list_pos[chosen_k - 1].tolist()
#     train_fns_neg = train_list_neg[chosen_k - 1].tolist()
#     validation_fns_pos = valid_list_pos[chosen_k - 1].tolist()
#     validation_fns_neg = valid_list_neg[chosen_k - 1].tolist()
#     validation_fns = valid_list_pos[chosen_k - 1].tolist() + valid_list_neg[chosen_k - 1].tolist()
#     test_fns_pos = test_list_pos[chosen_k - 1].tolist()
#     test_fns_neg = test_list_neg[chosen_k - 1].tolist()
#     test_fns = test_list_pos[chosen_k - 1].tolist() + test_list_neg[chosen_k - 1].tolist()
# 
#     # shuffle the results
#     random.shuffle(train_fns_pos)
#     random.shuffle(train_fns_neg)
#     random.shuffle(validation_fns)
#     random.shuffle(test_fns)
# 
#     # store results in a dict
#     result_dict = dict()
#     result_dict['train_fns_pos'] = train_fns_pos
#     result_dict['train_fns_neg'] = train_fns_neg
#     result_dict['validation_fns_neg'] = validation_fns_neg
#     result_dict['validation_fns_pos'] = validation_fns_pos
#     result_dict['validation_fns'] = validation_fns
#     result_dict['test_fns_neg'] = test_fns_neg
#     result_dict['test_fns_pos'] = test_fns_pos
#     result_dict['test_fns'] = test_fns
# 
#     return result_dict
# 
# def generate_kfold_v5b(df, k, chosen_k):
#     """
#     Generate a custom kfold validation containing train_pos, train_neg, validation and test for each round.
# 
#     :param df: pandas dataframe with phenotype information
#     :param k: number of kfold partitions
#     :param chosen_k: chosen k round to return the partitions data
#     :return: dictionary containing [train_pos, train_neg, validation, test] partitions from chosen K round
#     """
# 
#     # stratify samples and shuffle image files
#     # df['filename'] = args.image_location_prefix + df['FNAME_TFREC'].values  # store filename
#     df['filename'] = df['FNAME_TFREC'].values  # store filename
# 
#     n_bins = 50
# 
#     percentiles_split = np.linspace(0, 1, n_bins+1)
# 
#     df_pos = df[df['gender'] == 1].copy()  # male
#     df_neg = df[df['gender'] == 2].copy()  # female
#     quantiles_pos = df_pos['cl_tot'].quantile(percentiles_split).unique()
#     quantiles_neg = df_neg['cl_tot'].quantile(percentiles_split).unique()
# 
#     quantiles_pos = df_pos['cl_tot'].quantile(percentiles_split).unique()
#     quantiles_neg = df_neg['cl_tot'].quantile(percentiles_split).unique()
# 
#     conditions_pos, conditions_neg = [], []
#     for i in range(len(quantiles_pos) - 1):
#         if i < (len(quantiles_pos) - 2):
#             conditions_pos.append(df_pos['cl_tot'].between(quantiles_pos[i], quantiles_pos[i+1] - 0.001))
#         else:
#             conditions_pos.append(df_pos['cl_tot'].between(quantiles_pos[i], quantiles_pos[i+1]))
#     for i in range(len(quantiles_neg) - 1):
#         if i < (len(quantiles_neg) - 2):
#             conditions_neg.append(df_neg['cl_tot'].between(quantiles_neg[i], quantiles_neg[i+1] - 0.001))
#         else:
#             conditions_neg.append(df_neg['cl_tot'].between(quantiles_neg[i], quantiles_neg[i+1]))
#     bin_values_pos = np.arange(len(conditions_pos))
#     bin_values_neg = np.arange(len(conditions_neg))
# 
#     df_pos['cl_tot_bin'] = np.select(conditions_pos, bin_values_pos, default=0)
#     df_neg['cl_tot_bin'] = np.select(conditions_neg, bin_values_neg, default=0)
# 
#     fold_file_name = args.out_path + "kfold" + str(args.chosen_k) + "_examples.txt"
#     print_file(filename=args.log_file, text='\n----- Generate and return chosen Kfold partitions -----')
#     print_file(filename=args.log_file, text='  Number of folds = ' + str(k))
#     print_file(filename=args.log_file, text='  Chosen K partitions = ' + str(chosen_k))
#     # each item of theses lists is a partition of a given fold number
#     train_list_pos, train_list_neg = [], []
#     valid_list_pos, valid_list_neg = [], []
#     test_list_pos, test_list_neg = [], []
#     # create stratification
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
#     skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
# 
#     # positive classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_pos, df_pos['cl_tot_bin'])):
#         df_train, df_valid_test = df_pos.iloc[train_index], df_pos.iloc[valid_test_index]
#         train_list_pos.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         # print(df_valid_test.groupby('cl_tot_bin').agg(['count'])['filename'])
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_tot_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_pos.append(df_valid['filename'])
#                 test_list_pos.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\n---Positive partitions of K = ' + str(m+1), fold_file_name)
#             print_file('\ntrain_list_pos[' + str(m+1) + ']\n' + '\n'.join(train_list_pos[m]), fold_file_name)
#             print_file('\nvalid_list_pos[' + str(m+1) + ']\n' + '\n'.join(valid_list_pos[m]), fold_file_name)
#             print_file('\ntest_list_pos[' + str(m+1) + ']\n' + '\n'.join(test_list_pos[m]), fold_file_name)
# 
#     # negative classes stratification
#     for m, (train_index, valid_test_index) in enumerate(skf.split(df_neg, df_neg['cl_tot_bin'])):
#         df_train, df_valid_test = df_neg.iloc[train_index], df_neg.iloc[valid_test_index]
#         train_list_neg.append(df_train['filename'])
#         # split valid_test in validation and test partitions
#         valid_index, test_index = [], []
#         for n, (valid_index, test_index) in enumerate(skf2.split(df_valid_test, df_valid_test['cl_tot_bin'])):
#             if n == 1:
#                 df_valid, df_test = df_valid_test.iloc[valid_index], df_valid_test.iloc[test_index]
#                 valid_list_neg.append(df_valid['filename'])
#                 test_list_neg.append(df_test['filename'])
#         if m+1 == chosen_k:
#             print_file('\ntrain_list_neg[' + str(m+1) + ']\n' + '\n'.join(train_list_neg[m]), fold_file_name)
#             print_file('\nvalid_list_neg[' + str(m+1) + ']\n' + '\n'.join(valid_list_neg[m]), fold_file_name)
#             print_file('\ntest_list_neg[' + str(m+1) + ']\n' + '\n'.join(test_list_neg[m]), fold_file_name)
# 
#     # assign the chosen K partitions to resulting variables
#     train_fns_pos = train_list_pos[chosen_k - 1].tolist()
#     train_fns_neg = train_list_neg[chosen_k - 1].tolist()
#     validation_fns_pos = valid_list_pos[chosen_k - 1].tolist()
#     validation_fns_neg = valid_list_neg[chosen_k - 1].tolist()
#     validation_fns = valid_list_pos[chosen_k - 1].tolist() + valid_list_neg[chosen_k - 1].tolist()
#     test_fns_pos = test_list_pos[chosen_k - 1].tolist()
#     test_fns_neg = test_list_neg[chosen_k - 1].tolist()
#     test_fns = test_list_pos[chosen_k - 1].tolist() + test_list_neg[chosen_k - 1].tolist()
# 
#     # shuffle the results
#     random.shuffle(train_fns_pos)
#     random.shuffle(train_fns_neg)
#     random.shuffle(validation_fns)
#     random.shuffle(test_fns)
# 
#     # store results in a dict
#     result_dict = dict()
#     result_dict['train_fns_pos'] = train_fns_pos
#     result_dict['train_fns_neg'] = train_fns_neg
#     result_dict['validation_fns_neg'] = validation_fns_neg
#     result_dict['validation_fns_pos'] = validation_fns_pos
#     result_dict['validation_fns'] = validation_fns
#     result_dict['test_fns_neg'] = test_fns_neg
#     result_dict['test_fns_pos'] = test_fns_pos
#     result_dict['test_fns'] = test_fns
# 
#     return result_dict
