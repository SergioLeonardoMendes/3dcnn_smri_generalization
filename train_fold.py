# import tensorflow_addons as tfa
# from swa.tfkeras import SWA
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_dataset import get_training_dataset
import pandas as pd
import matplotlib.pyplot as plt
from predict import *
from utils import *
from models import *

# CONSTANT DECLARATIONS
IMAGE_SIZE_OUT = (128, 128, 128, 2)

def display_one_brain(image, title, subplot, color):
    def display_one_slice(slice, subp2, tit):
        plt.subplot(subplot[0], subplot[1], subp2)
        plt.axis('off')
        plt.imshow(np.rot90(slice, 2), cmap='gray')
        plt.title(tit, fontsize=16, color=color)

    subp2 = subplot[2]
    tissues = image.shape[3]
    for tissue in range(tissues):
        tit = title+'t'+str(tissue)+'x'
        display_one_slice(image[60, :, :, tissue], subp2, tit)
        subp2 += 1
        tit = title+'t'+str(tissue)+'y'
        display_one_slice(image[:, 72, :, tissue], subp2, tit)
        subp2 += 1
        tit = title+'t'+str(tissue)+'z'
        display_one_slice(image[:, :, 60, tissue], subp2, tit)
        subp2 += 1

def display_many_brains(images, classes, title_colors=None):
    tissues = images[0].shape[3]
    plt.figure(figsize=(4.5*tissues, 14))

    for i in range(8):
        color = 'black' if title_colors is None else title_colors[i]
        display_one_brain(images[i], 'cer'+str(i), [8, 3*tissues, 1+i*3*tissues], color)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    print_file(filename=ARGS.log_file, text="  " + ARGS.out_path + 'brain_examples.png')
    plt.savefig(ARGS.out_path+'brain_examples.png', bbox_inches='tight')

class LogMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LogMetricsCallback, self).__init__()
        self.best_val_loss = np.inf

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = tf.timestamp() - self.start_time
        print_file(filename=ARGS.log_file,
                   text="  Epoch %04d: time=%ds loss=%.2f val_loss=%.2f mae=%.2f val_mae=%.2f"
                        % (epoch, elapsed_time, logs['loss'], logs['val_loss'],  # logs['auc'], logs['val_auc'])
                            logs['mean_absolute_error'], logs['val_mean_absolute_error'])
        )
        if logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            print_file(filename=ARGS.log_file, text="  A better model was encountered. MODEL SAVED!")

# configure and return the callbacks used during training
def get_callbacks():

    best_val_loss = np.inf

    mcp_save_best = ModelCheckpoint(ARGS.out_path + 'checkpoints/best_model',
                                    save_weights_only=False, save_best_only=True,
                                    # monitor='val_auc', verbose=1, mode='max') #val_loss
                                    monitor='val_loss', verbose=1, mode='min')
    mcp_save_last = ModelCheckpoint(ARGS.out_path + 'checkpoints/last_model',
                                    save_weights_only=False, verbose=1)

    # mcp_save_avg_best = tfa.callbacks.AverageModelCheckpoint(filepath=ARGS.out_path + 'checkpoints/best_avg',
    #                                                     save_weights_only=False, save_best_only=True,
    #                                                     monitor='val_loss', verbose=1, mode='min',
    #                                                     update_weights=True)
    # # define swa callback
    # swa = SWA(start_epoch=15,
    #           lr_schedule='constant',
    #           swa_lr=0.0005,
    #           verbose=1,
    #           batch_size=14)

    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=50, mode='max') #val_loss
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min')
    tensorboard = tf.keras.callbacks.TensorBoard(ARGS.tboard_path, update_freq=1, histogram_freq=1)
    log_metrics = LogMetricsCallback()

    # lr decay: reduce LR by 'factor' after 'patience' iterations without performance improvement
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-5)

    return [log_metrics, mcp_save_best, early_stopping, tensorboard]

def count_data_items(filenames):
    # the number examples per filename (i.e. 1)
    n = len(filenames) * 1
    return n

def get_model_summary(model: tf.keras.Model) -> str:
    string_list = []
    model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)

def main(arguments):

    global ARGS
    ARGS = arguments

    create_dirs(ARGS)
    print_configs(ARGS)
    enable_determinism(ARGS)
    config_gpus(ARGS)

    print_file(filename=ARGS.log_file, text="\n----- Phenotypes csv location -----\n")
    print_file(filename=ARGS.log_file, text="  " + ARGS.csv_phenotypes)

    # load train validation test examples
    train_examples = pd.read_csv(ARGS.train_csv_path).values
    valid_examples = pd.read_csv(ARGS.valid_csv_path).values

    print_file(filename=ARGS.log_file, text="\n----- Create tensorflow data generators -----\n")
    training_dataset = get_training_dataset(ARGS, train_examples)
    validation_dataset = get_assess_dataset(ARGS, valid_examples)

    print_file(filename=ARGS.log_file, text="\n----- Generate brain image examples -----\n")
    print_file(filename=ARGS.log_file, text="")
    iterator = get_dataset_iterator(training_dataset, 8)
    inputs, labels = next(iterator) # get first batch of images
    display_many_brains(inputs['image'], labels['output1'])
    print_file(filename=ARGS.log_file, text="")

    print_file(filename=ARGS.log_file, text="\n----- Create and compile model -----")
    optimizer = tf.keras.optimizers.Adam(learning_rate=ARGS.lr, beta_1=ARGS.beta_1, beta_2=ARGS.beta_2)
    # sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # swa_optimizer = tfa.optimizers.MovingAverage(optimizer)
    # swa_optimizer = tfa.optimizers.SWA(optimizer)

    # prepare model params
    losses = eval(ARGS.model_losses)
    metrics = eval(ARGS.model_metrics)
    initial_bias = eval(ARGS.model_out_bias)

    # ['AUC','Precision','Recall', 'BinaryAccuracy','TruePositives','FalsePositives',
    # 'TrueNegatives','FalseNegatives']
    # output bias initialization example
    # ini_bias_sex = np.log([422 / 318])  # num examples pos=422 and neg=318
    # ini_bias_age = 9.92  # age mean
    # initial_bias = None

    # Use available GPUs through Nvidia NCCL cross device communication
    strategy = tf.distribute.MirroredStrategy()
    print_file(filename=ARGS.log_file, text="  Distribute strategy: " + str(strategy.__class__.__name__))
    with strategy.scope():
        # input shape of the model=((x_axis, y_axis, z_axix, alpha), output_classes)
        model_str = ARGS.model_arch + '(IMAGE_SIZE_OUT, output_bias=initial_bias, ' \
                                      'kern_reg_l2=ARGS.kern_reg_l2, dropout=ARGS.dropout)'
        model = eval(model_str)
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)  # loss_weights=[0.5, 1, 0.5],

    print_file(filename=ARGS.log_file, text="\n----- Print model architecture -----")
    model_print = eval(model_str)
    print_file(filename=ARGS.log_file, text="  " + ARGS.out_path + "model_design.txt")
    print_file(text=get_model_summary(model_print), filename=ARGS.out_path + "model_design.txt")

    print_file(filename=ARGS.log_file, text="\n----- Dataset info -----")
    n_train = count_data_items(train_examples)
    n_valid = count_data_items(valid_examples)

    train_steps = (n_train // ARGS.batch_size) + 1
    print_file(filename=ARGS.log_file, text="  TRAINING IMAGES: "+str(n_train) + ", STEPS PER EPOCH: "+str(train_steps))
    print_file(filename=ARGS.log_file, text="  VALIDATION IMAGES: "+str(n_valid))

    print_file(filename=ARGS.log_file, text="\n----- Training model -----")
    print_file(filename=ARGS.log_file, text="")
    # generate callbacks
    callbacks = get_callbacks()
    # train the model
    history = model.fit(training_dataset, validation_data=validation_dataset,
                        callbacks=callbacks, steps_per_epoch=train_steps, epochs=ARGS.n_epochs)
    print_file(filename=ARGS.log_file, text="")


# if __name__ == '__main__':
#     """
#     Usage example:
#         docker exec -it sgotf2ctn /usr/bin/python /project/sources/train_fold.py
#             --project_config_path /project/sources/config/INPD/
#             --train_csv_path /project/data/INPD/train_valid_test/kfold10-gender0cl_tot10-seed88/k01.train.csv
#             --valid_csv_path /project/data/INPD/train_valid_test/kfold10-gender0cl_tot10-seed88/k01.valid.csv
#             --test_csv_path /project/data/INPD/train_valid_test/kfold10-gender0cl_tot10-seed88/k01.test.csv
#             --OTHER_PARAMETERS(see default_parameters.json)
#     """
#
#     shell_args = set_config_train_fold()
#     main(shell_args)
