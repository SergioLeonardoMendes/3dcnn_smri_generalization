import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import SimpleITK as sitk
from scipy.stats import ttest_ind, t, pearsonr
from scipy import ndimage
from sklearn.metrics import roc_auc_score, auc
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import glob, re, os
from utils import print_file


results_prefix = '/project/output/ABCD_create_tfrecs/'
results_logfile = results_prefix + 'output.log'
os.makedirs(results_prefix, exist_ok=True)


# Pandas display options
pd.set_option('display.max_rows', 1200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 100)


# find files and store infos in df_files
files = glob.glob('/project/data/ABCD/images/' + '/**/smwc1*.nii', recursive=True)
df_files = pd.DataFrame(files, columns=['FNAME_C1'])
df_files['FNAME_C2'] = df_files['FNAME_C1'].str.replace('smwc1','smwc2')
df_files['FNAME_C3'] = df_files['FNAME_C1'].str.replace('smwc1','smwc3')
df_files['FNAME_TFREC'] = df_files['FNAME_C1'].str.replace('smwc1','')
df_files['FNAME_TFREC'] = df_files['FNAME_TFREC'].str.replace('nii','tfrec')
df_files['FNAME_TFREC'] = df_files['FNAME_TFREC'].str.replace('/images/','/vbmspm12/')
df_files['subjectkey'] = df_files['FNAME_C1'].str[30:30+15]
df_files['subjectkey'] = df_files['subjectkey'].str.replace('NDAR','NDAR_')
# load MRI infos in df_mri
df_mri = pd.read_csv('/project/data/ABCD/phenotypes/abcd_mri01.txt', sep='\t', header=0, skiprows=[1])
df_mri = df_mri[df_mri['mri_info_deviceserialnumber'].notnull()]
# save encoded scanner device serial number
le = preprocessing.LabelEncoder()
le.fit(df_mri['mri_info_deviceserialnumber'].values)
df_mri['scanner_serialnumber_bin'] = le.transform(df_mri['mri_info_deviceserialnumber'].values)
# load cbcl infos in df_cbcl
df_cbcl = pd.read_csv('/project/data/ABCD/phenotypes/abcd_cbcls01.txt', sep='\t', header=0, skiprows=[1])
df_cbcl['sex_bin'] = df_cbcl['sex'].apply(lambda x: 0 if x == 'F' else 1) # map gender variable
# merge infos in df
df = pd.merge(df_cbcl, df_files, on='subjectkey')
df = pd.merge(df, df_mri, on='subjectkey', how='left')
# exclude null rows for cbcl total and interview age
condition = (
    df['eventname_x']=='baseline_year_1_arm_1') & (
    df['eventname_y']=='baseline_year_1_arm_1') & (
    df['cbcl_scr_syn_totprob_r'].notna()) & (
    df['interview_age_x'].notna())
df = df[condition]
df['interview_age'] = df['interview_age_x']
df['FNAME_TFREC'] = df['FNAME_TFREC'].str.replace('/ABCD/','/tfrec/ABCD/')
# save dataframe to csv
df.to_csv('/project/data/ABCD/phenotypics.csv', index=False, na_rep='')
# load dataframe
df = pd.read_csv('/project/data/ABCD/phenotypics.csv')
# select columns to save tf_records
basic_columns = ['subjectkey','interview_age','sex_bin']
cbcl_columns = ['cbcl_scr_syn_anxdep_r', 'cbcl_scr_syn_anxdep_t',
    'cbcl_scr_syn_withdep_r', 'cbcl_scr_syn_withdep_t',
    'cbcl_scr_syn_somatic_r', 'cbcl_scr_syn_somatic_t',
    'cbcl_scr_syn_social_r', 'cbcl_scr_syn_social_t',
    'cbcl_scr_syn_thought_r', 'cbcl_scr_syn_thought_t',
    'cbcl_scr_syn_attention_r', 'cbcl_scr_syn_attention_t',
    'cbcl_scr_syn_rulebreak_r', 'cbcl_scr_syn_rulebreak_t',
    'cbcl_scr_syn_aggressive_r', 'cbcl_scr_syn_aggressive_t',
    'cbcl_scr_syn_internal_r', 'cbcl_scr_syn_internal_t',
    'cbcl_scr_syn_external_r', 'cbcl_scr_syn_external_t',
    'cbcl_scr_syn_totprob_r', 'cbcl_scr_syn_totprob_t',
    'cbcl_scr_dsm5_depress_r', 'cbcl_scr_dsm5_depress_t',
    'cbcl_scr_dsm5_anxdisord_r', 'cbcl_scr_dsm5_anxdisord_t',
    'cbcl_scr_dsm5_somaticpr_r', 'cbcl_scr_dsm5_somaticpr_t',
    'cbcl_scr_dsm5_adhd_r', 'cbcl_scr_dsm5_adhd_t',
    'cbcl_scr_dsm5_opposit_r','cbcl_scr_dsm5_opposit_t',
    'cbcl_scr_dsm5_conduct_r','cbcl_scr_dsm5_conduct_t',
    'cbcl_scr_07_sct_r', 'cbcl_scr_07_sct_t',
    'cbcl_scr_07_ocd_r','cbcl_scr_07_ocd_t',
    'cbcl_scr_07_stress_r', 'cbcl_scr_07_stress_t']


lb = preprocessing.LabelBinarizer()
X_AGE = df['interview_age'].values.reshape(-1, 1)
X_SEX = df['sex_bin'].values.reshape(-1, 1)
X_SCANNER = lb.fit(df['scanner_serialnumber_bin'].values).transform(
    df['scanner_serialnumber_bin'].values)
y = df['cbcl_scr_syn_totprob_r'].values
pred_cbcl_age = LinearRegression().fit(X_AGE, y).predict(X_AGE)
pred_cbcl_sex = LinearRegression().fit(X_SEX, y).predict(X_SEX)
pred_cbcl_scanner = LinearRegression().fit(X_SCANNER, y).predict(X_SCANNER)
print_file(filename=results_logfile, text='\n##### STATISTICAL IMPACT OF COVARIATES ON CBCL TOTAL #####\n')
print_file(filename=results_logfile, text='AGE (pearson-r,p-value) = '+
                                          str(pearsonr(df['cbcl_scr_syn_totprob_r'].values, pred_cbcl_age)))
print_file(filename=results_logfile, text='SEX (pearson-r,p-value) = '+
                                          str(pearsonr(df['cbcl_scr_syn_totprob_r'].values, pred_cbcl_sex)))
print_file(filename=results_logfile, text='SCANNER (pearson-r,p-value) = '+
                                          str(pearsonr(df['cbcl_scr_syn_totprob_r'].values, pred_cbcl_scanner)))
print_file(filename=results_logfile, text='\n##########################################################\n')


# function to load the .nii files, store and return a 4d numpy array
def load_image(subject_id, dataframe):
    # read the .nii image containing the volume with SimpleITK:
    sitk_t1c1 = sitk.ReadImage(dataframe[dataframe['subjectkey'] == subject_id]['FNAME_C1'].values[0])
    sitk_t1c2 = sitk.ReadImage(dataframe[dataframe['subjectkey'] == subject_id]['FNAME_C2'].values[0])
    sitk_t1c3 = sitk.ReadImage(dataframe[dataframe['subjectkey'] == subject_id]['FNAME_C3'].values[0])
    # access the numpy array:
    t1c1 = sitk.GetArrayFromImage(sitk_t1c1)
    t1c2 = sitk.GetArrayFromImage(sitk_t1c2)
    t1c3 = sitk.GetArrayFromImage(sitk_t1c3)
    # create a 4d array and save the MRI content into t1
    img4d = np.ndarray(shape=(t1c1.shape[0], t1c1.shape[1], t1c1.shape[2], 3), dtype=np.float32)
    img4d[:, :, :, 0] = t1c1
    img4d[:, :, :, 1] = t1c2
    img4d[:, :, :, 2] = t1c3
    return np.array(img4d)
# function to load the .nii files, store and return a 4d numpy array
def load_image_raw(subject_id, dataframe):
    # read the .nii image containing the volume with SimpleITK:
    sitk_t1full = sitk.ReadImage(dataframe[dataframe['subjectkey'] == subject_id]['FNAME_HDBET'].values[0])
    # access the numpy array:
    t1full = sitk.GetArrayFromImage(sitk_t1full)
    # create a 4d array and save the MRI content into t1
    img3d = np.ndarray(shape=(t1full.shape[0], t1full.shape[1], t1full.shape[2]), dtype=np.float32)
    img3d = t1full
    return np.array(img3d)


file_list = df['subjectkey'].values
t1 = load_image(file_list[0],df)
x_mid = int(t1.shape[0]/2)
y_mid = int(t1.shape[1]/2)
z_mid = int(t1.shape[2]/2)
fig_number = 100
for n in range(1): # for n in range((len(file_list)//10)+1):
    plt.figure(fig_number)
    fig_number += 1
    # create a subplot
    f, ax = plt.subplots(10, 6, figsize=(13,25), tight_layout=True)
    f.patch.set_facecolor('white')
    # set all subplot axis to off
    [axi.set_axis_off() for axi in ax.ravel()]
    for i in range(10):
        if (n*10 + i) < len(file_list):
            t1 = load_image(file_list[n*10 + i],df)
            # display gray matter
            ax[i,0].imshow(np.rot90((t1[x_mid,:,:,0]),2))
            ax[i,0].text(2, 1, str(file_list[n*10 + i]), fontsize=10, #color='red',
                        bbox=dict(facecolor='white'))
            ax[i,2].imshow(np.rot90((t1[:,y_mid,:,0]),2))
            ax[i,4].imshow(np.rot90((t1[:,:,z_mid,0]),2))
            # display white matter
            ax[i,1].imshow(np.rot90((t1[x_mid,:,:,1]),2))
            ax[i,3].imshow(np.rot90((t1[:,y_mid,:,1]),2))
            ax[i,5].imshow(np.rot90((t1[:,:,z_mid,1]),2))
    plt.savefig(results_prefix + 'vbm_examples_' + str(n) + '.png')


print_file(filename=results_logfile, text='##############################')
print_file(filename=results_logfile, text='Male:' + str(df[df['sex_bin']==1]['subjectkey'].count()))
print_file(filename=results_logfile, text='Female:' + str(df[df['sex_bin']==0]['subjectkey'].count()))
print_file(filename=results_logfile, text='Age(min):' + str(df['interview_age'].min()))
print_file(filename=results_logfile, text='Age(max):' + str(df['interview_age'].max()))
print_file(filename=results_logfile, text='Age(mean):' + str(df['interview_age'].mean()))
print_file(filename=results_logfile, text='Age(sd):' + str(df['interview_age'].std()))
print_file(filename=results_logfile, text='Cbcltot_r(min):' + str(df['cbcl_scr_syn_totprob_r'].min()))
print_file(filename=results_logfile, text='Cbcltot_r(max):' + str(df['cbcl_scr_syn_totprob_r'].max()))
print_file(filename=results_logfile, text='Cbcltot_r(mean):' + str(df['cbcl_scr_syn_totprob_r'].mean()))
print_file(filename=results_logfile, text='Cbcltot_r(sd):' + str(df['cbcl_scr_syn_totprob_r'].std()))
print_file(filename=results_logfile, text='Cbcltot_t(min):' + str(df['cbcl_scr_syn_totprob_t'].min()))
print_file(filename=results_logfile, text='Cbcltot_t(max):' + str(df['cbcl_scr_syn_totprob_t'].max()))
print_file(filename=results_logfile, text='Cbcltot_t(mean):' + str(df['cbcl_scr_syn_totprob_t'].mean()))
print_file(filename=results_logfile, text='Cbcltot_t(sd):' + str(df['cbcl_scr_syn_totprob_t'].std()))
print_file(filename=results_logfile, text='##############################\n')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _float_image(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


image_sum = np.zeros(shape=(121, 145, 121))
image_count = 0
for index, meta_data in df.iterrows():
    # load the image into a 4d numpy array
    image = load_image(meta_data['subjectkey'],df)
    # increment count and image_sum array
    image_count += 1
    image_sum += np.sum(image, axis=3)
    #print(meta_data['SCANDIR_ID'],image.shape, 'sum='+str(image_sum.shape))
print_file(filename=results_logfile, text='image_count: '+str(image_count) +
                                          ' image_sum.shape ='+str(image_sum.shape))


#create average image
image_avg = image_sum / image_count
#create mask
limiar_mask = 0.05
image_mask = (image_avg > 0) & (image_avg < limiar_mask)
#create noise cleaner 3d and 4d
noise_cleaner = ~image_mask
noise_cleaner4d = np.repeat(noise_cleaner[:, :, :, np.newaxis], 3, axis=3)
#noise_cleaner4d = (noise_cleaner[...,np.newaxis]+np.zeros(3)[np.newaxis,np.newaxis,np.newaxis,:])
# create a background mask with 1's
image_bg1_mask = (noise_cleaner * image_avg == 0)
image_bg1_mask4d = np.repeat(image_bg1_mask[:, :, :, np.newaxis], 3, axis=3)
print_file(filename=results_logfile, text='Saving background mask...')
np.save('/project/data/ABCD/ABCD_bg1_mask4d005.npy',image_bg1_mask4d)
np.save(results_prefix+'ABCD_bg1_mask4d005.npy',image_bg1_mask4d)
print_file(filename=results_logfile, text='Loading background mask...\n')
image_bg1_mask4d=np.load(results_prefix + 'ABCD_bg1_mask4d005.npy')
# show noise cleaner filter
# plt.imshow((noise_cleaner4d[61,:,:,2]).astype(np.uint8))
# plt.imshow((image_bg1_mask4d[61,:,:,2]).astype(np.uint8))
# print_file(filename=results_logfile, text='##### DONE! #####\n')

# read the .nii image containing the volume with SimpleITK:
sitk_t1c1 = sitk.ReadImage(df['FNAME_C1'][0])
sitk_t1c2 = sitk.ReadImage(df['FNAME_C2'][0])
sitk_t1c3 = sitk.ReadImage(df['FNAME_C3'][0])
# access the numpy array:
t1c1 = sitk.GetArrayFromImage(sitk_t1c1)
t1c2 = sitk.GetArrayFromImage(sitk_t1c2)
t1c3 = sitk.GetArrayFromImage(sitk_t1c3)
# create a 4d array from MRI content into t1
t1 = np.ndarray(shape=(t1c1.shape[0],t1c1.shape[1],t1c1.shape[2],3))
t1[:,:,:,0] = t1c1 * noise_cleaner
t1[:,:,:,1] = t1c2 * noise_cleaner
t1[:,:,:,2] = t1c3 * noise_cleaner
# clean image mean
t1_sum = (t1c1 + t1c2 + t1c3) * noise_cleaner
# create a -1 background mask
t1_sum_bg1_mask = -1 * (t1_sum == 0)
#create mask
limiar = 0.03
t1mask = (t1 > 0) & (t1 < limiar)
t1mask_sum = (t1_sum > 0) & (t1_sum < limiar)
cmap = plt.get_cmap('gray')
t1rgba, t1, t1rgba_sum, t1_sum = cmap(t1), cmap(t1), cmap(t1_sum), cmap(t1_sum) #cmap(np.zeros(shape=t1_sum.shape))
t1rgba[:,:,:,:,0] = t1mask / 2
t1[:,:,:,:,0] = t1[:,:,:,:,0] / 2
t1rgba_sum[:,:,:,0] = t1mask_sum / 2
t1_sum[:,:,:,0] = t1_sum[:,:,:,0] / 2
plt.figure(71)
# create a subplot
f, ax = plt.subplots(5, 6, figsize=(13,9), tight_layout=True)
# set all subplot axis to off
[axi.set_axis_off() for axi in ax.ravel()]
# set title according to each example
title = 'Gray, White and CSF'
x_mid = int(t1.shape[0]/2)
y_mid = int(t1.shape[1]/2)
z_mid = int(t1.shape[2]/2)
# display gray matter
ax[0,0].imshow(np.rot90((t1[x_mid,:,:,0]*255).astype(np.uint8),2))
ax[0,2].imshow(np.rot90((t1[:,y_mid,:,0]*255).astype(np.uint8),2))
ax[0,4].imshow(np.rot90((t1[:,:,z_mid,0]*255).astype(np.uint8),2))
# display white matter
ax[1,0].imshow(np.rot90((t1[x_mid,:,:,1]*255).astype(np.uint8),2))
ax[1,2].imshow(np.rot90((t1[:,y_mid,:,1]*255).astype(np.uint8),2))
ax[1,4].imshow(np.rot90((t1[:,:,z_mid,1]*255).astype(np.uint8),2))
# display csf
ax[2,0].imshow(np.rot90((t1[x_mid,:,:,2]*255).astype(np.uint8),2))
ax[2,2].imshow(np.rot90((t1[:,y_mid,:,2]*255).astype(np.uint8),2))
ax[2,4].imshow(np.rot90((t1[:,:,z_mid,2]*255).astype(np.uint8),2))
# display gray matter
ax[0,1].imshow(np.rot90((t1rgba[x_mid,:,:,0]*255).astype(np.uint8),2))
ax[0,3].imshow(np.rot90((t1rgba[:,y_mid,:,0]*255).astype(np.uint8),2))
ax[0,5].imshow(np.rot90((t1rgba[:,:,z_mid,0]*255).astype(np.uint8),2))
# display white matter
ax[1,1].imshow(np.rot90((t1rgba[x_mid,:,:,1]*255).astype(np.uint8),2))
ax[1,3].imshow(np.rot90((t1rgba[:,y_mid,:,1]*255).astype(np.uint8),2))
ax[1,5].imshow(np.rot90((t1rgba[:,:,z_mid,1]*255).astype(np.uint8),2))
# display csf
ax[2,1].imshow(np.rot90((t1rgba[x_mid,:,:,2]*255).astype(np.uint8),2))
ax[2,3].imshow(np.rot90((t1rgba[:,y_mid,:,2]*255).astype(np.uint8),2))
ax[2,5].imshow(np.rot90((t1rgba[:,:,z_mid,2]*255).astype(np.uint8),2))
# display sum
ax[3,0].imshow(np.rot90((t1_sum[x_mid,:,:]*255).astype(np.uint8),2))
ax[3,2].imshow(np.rot90((t1_sum[:,y_mid,:]*255).astype(np.uint8),2))
ax[3,4].imshow(np.rot90((t1_sum[:,:,z_mid]*255).astype(np.uint8),2))
ax[3,1].imshow(np.rot90((t1rgba_sum[x_mid,:,:]*255).astype(np.uint8),2))
ax[3,3].imshow(np.rot90((t1rgba_sum[:,y_mid,:]*255).astype(np.uint8),2))
ax[3,5].imshow(np.rot90((t1rgba_sum[:,:,z_mid]*255).astype(np.uint8),2))
ax[4,0].imshow(np.rot90((t1_sum[x_mid,:,:]*255).astype(np.uint8),2))
ax[4,2].imshow(np.rot90((t1_sum[:,y_mid,:]*255).astype(np.uint8),2))
ax[4,4].imshow(np.rot90((t1_sum[:,:,z_mid]*255).astype(np.uint8),2))
ax[4,1].imshow(np.rot90((t1_sum_bg1_mask[x_mid,:,:]*255).astype(np.uint8),2))
ax[4,3].imshow(np.rot90((t1_sum_bg1_mask[:,y_mid,:]*255).astype(np.uint8),2))
ax[4,5].imshow(np.rot90((t1_sum_bg1_mask[:,:,z_mid]*255).astype(np.uint8),2))
plt.savefig(results_prefix + 'filter_mask.png')


import matplotlib.animation as animation
VIDEO_FILEPATH = results_prefix + 'noise_remove_1-5.mp4'
# transform images to rgba
cmap = plt.get_cmap('gray')
image_rgba, image_avg_rgba = cmap(image_avg), cmap(image_avg)
# assign the mask to the red channel and normalize color of image
# image_rgba[:,:,:,0] = image_mask / 2
image_rgba[:, :, :, 0] = -0.5 * t1_sum_bg1_mask / 2
image_avg_rgba[:, :, :, 0] = image_avg_rgba[:, :, :, 0] / 2
# initialize fig and img variables
fig = plt.figure(51, figsize=(4 * 3.91, 4 * 2.92), dpi=100)
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.axis('off')
ims = []
for i in range(145):
    frame2d = np.zeros(shape=(292, 391, 4))
    if (i < 121):
        frame2d[0:145, 0:121, :] = np.rot90(image_avg_rgba[i, :, :], 2)
        frame2d[12:133, 123:244, :] = np.rot90(image_avg_rgba[:, i, :], 2)
        frame2d[12:133, 246:391, :] = np.rot90(image_avg_rgba[:, :, i], 2)
        frame2d[147 + 0:147 + 145, 0:121, :] = np.rot90(image_rgba[i, :, :], 2)
        frame2d[147 + 12:147 + 133, 123:244, :] = np.rot90(image_rgba[:, i, :], 2)
        frame2d[147 + 12:147 + 133, 246:391, :] = np.rot90(image_rgba[:, :, i], 2)
    else:
        frame2d[12:133, 123:244, :] = np.rot90(image_avg_rgba[:, i, :], 2)
        frame2d[147 + 12:147 + 133, 123:244, :] = np.rot90(image_rgba[:, i, :], 2)
    ims.append([plt.imshow(frame2d, animated=True)])
# create the animation
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
# write the video file to disk
ani.save(VIDEO_FILEPATH)


# path to save the TFRecords file
tfrec_filename = df['FNAME_TFREC'].values[0]
# create directory if not exists
dir_tfrec = os.path.dirname(tfrec_filename)
os.makedirs(dir_tfrec, exist_ok=True)
# open the file
writer = tf.io.TFRecordWriter(tfrec_filename)
# iterate on dataframe saving rec_per_file records per file
rec_per_files = 1
for index, meta_data in df.iterrows():
    # file number, with 4 images in each
    file_number = (index // rec_per_files) + 1
    # close the previous writer and open a new one every rec_per_file records
    if (index % rec_per_files == 0):
        if (index % 20 == 0):
            print_file(filename=results_logfile, text='Record:' + str(index))
        # new path to save the TFRecords file
        tfrec_filename = meta_data['FNAME_TFREC']
        # create directory if not exists
        dir_tfrec = os.path.dirname(tfrec_filename)
        os.makedirs(dir_tfrec, exist_ok=True)
        # close the previous writer
        writer.close()
        # open the file
        writer = tf.io.TFRecordWriter(tfrec_filename)
        # load the image into a 4d numpy array
    image = load_image(meta_data['subjectkey'], df)
    #     # apply noise cleaner to image
    #     image = image * noise_cleaner4d
    # create dictionary of features
    dict_features = {}
    for colname, dtype in df[basic_columns].dtypes.items():
        key = 'info/' + colname
        if dtype == 'object':
            value = _bytes_feature(bytes(meta_data[colname], 'utf8'))
        elif dtype == 'float64':
            value = _float_feature(meta_data[colname])
        elif dtype == 'int64':
            value = _int64_feature(meta_data[colname])
        dict_features[key] = value
    if ~np.isnan(meta_data['cbcl_scr_syn_totprob_r']):
        for colname, dtype in df[cbcl_columns].dtypes.items():
            key = 'info/' + colname
            if dtype == 'object':
                value = _bytes_feature(bytes(meta_data[colname], 'utf8'))
            elif dtype == 'float64':
                value = _float_feature(meta_data[colname])
            elif dtype == 'int64':
                value = _int64_feature(meta_data[colname])
            dict_features[key] = value
    dict_features['image'] = _float_image(image.ravel())
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=dict_features))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
# close the last writer
writer.close()


AUTO = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [121, 145, 121, 3]
# batch_size = 16 * tpu_strategy.num_replicas_in_sync
batch_size = 16
validation_split = 0.10
filenames = df[df['cbcl_scr_syn_totprob_r'].notnull()]['FNAME_TFREC'].values
split = len(filenames) - int(len(filenames) * validation_split)
train_fns = filenames[:split]
validation_fns = filenames[split:]
def parse_tfrecord(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = {'image': tf.io.FixedLenFeature(IMAGE_SIZE, tf.float32),
                'info/cbcl_scr_syn_totprob_r': tf.io.FixedLenFeature([], tf.float32)}
    example = tf.io.parse_single_example(serialized_example, features)
    # one_hot_dxgroup = tf.reshape(tf.sparse.to_dense(example['info/DX_GROUP']), [2])
    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return example['image'], example['info/cbcl_scr_syn_totprob_r']  # , one_hot_dxgroup
def load_dataset(filenames):
    # Read from TFRecords. For optimal performance, we interleave reads from multiple files.
    records = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    return records.map(parse_tfrecord, num_parallel_calls=AUTO)
def get_training_dataset():
    dataset = load_dataset(train_fns)
    # Prefetch the next batch while training (autotune prefetch buffer size).
    return dataset.repeat().shuffle(100).batch(batch_size).prefetch(AUTO)
training_dataset = get_training_dataset()
validation_dataset = load_dataset(validation_fns).batch(batch_size).prefetch(AUTO)
def display_one_brain(image, title, subplot, color):
    def display_one_slice(slice, subp2, tit):
        plt.subplot(subplot[0], subplot[1], subp2)
        plt.axis('off')
        plt.imshow(ndimage.rotate((slice * 255).astype(np.uint8), 180, reshape=True), cmap='gray')
        plt.title(tit, fontsize=16, color=color)
    subp2 = subplot[2]
    for tissue in range(3):
        tit = title + 'tec' + str(tissue) + 'x'
        display_one_slice(image[61, :, :, tissue], subp2, tit)
        subp2 += 1
        tit = title + 'tec' + str(tissue) + 'y'
        display_one_slice(image[:, 72, :, tissue], subp2, tit)
        subp2 += 1
        tit = title + 'tec' + str(tissue) + 'z'
        display_one_slice(image[:, :, 61, tissue], subp2, tit)
        subp2 += 1
# If model is provided, use it to generate predictions.
def display_many_brains(images, classes, title_colors=None):
    subplot = [10, 9, 1]
    plt.figure(figsize=(15, 16))
    for i in range(10):
        color = 'black' if title_colors is None else title_colors[i]
        display_one_brain(images[i], str(classes[i]) + ':' + 'cer' + str(i), [10, 9, 1 + i * 9], color)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.savefig(results_prefix + 'tfrec_brain_images.png')
def get_dataset_iterator(dataset, n_examples):
    return dataset.unbatch().batch(n_examples).as_numpy_iterator()
iterator = get_dataset_iterator(training_dataset, 10)
# Re-run this cell to show a new batch of images
CLASSES = ['Control', 'Autism']
images, classes = next(iterator)
# class_idxs = np.argmax(classes, axis=-1) # transform from one-hot array to class number
# labels = [CLASSES[idx] for idx in class_idxs]
# display_nine_flowers(images, labels)
display_many_brains(images, classes)
