import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, \
                      MaxPool3D, ZeroPadding3D, BatchNormalization, \
                      AvgPool3D, UpSampling3D, Concatenate, GlobalMaxPool3D, ReLU
from tensorflow.keras import activations, regularizers
from coord_conv import CoordinateChannel3D


def cnn3d(input_dim, output_bias=None, kern_reg_l2=0.001, dropout=0.5):

    input_1 = keras.Input(shape=input_dim, name='image')

    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1')(input_1)
    x = BatchNormalization(name='bn_1')(x)
    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_1')(x)

    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_2')(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5')(x)
    x = BatchNormalization(name='bn_5')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_6')(x)
    x = BatchNormalization(name='bn_6')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_3')(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7')(x)
    x = BatchNormalization(name='bn_7')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8')(x)
    x = BatchNormalization(name='bn_8')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4')(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7b')(x)
    x = BatchNormalization(name='bn_7b')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8b')(x)
    x = BatchNormalization(name='bn_8b')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4b')(x)

    x = Flatten(name='flt_1')(x)  # LEO MOD
    x = BatchNormalization(name='bn_9')(x)
    x = Dropout(dropout, name='dpt_1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(kern_reg_l2), name='d_1')(x)  # 'MtlBestTry01'
    # x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(kern_reg_l2), name='d_1')(x) # 'MtlBestTry03'

    out_common = BatchNormalization(name='out_common')(x)

    # encoder_image = keras.Model(inputs=input_1, outputs=out_common,
    #                       name='encoder_image')

    # print('Out_0:',tf.shape(out_0))
    # print('Out_common:',tf.shape(out_common))

    # predict output1
    x3 = Dense(4, activation='relu',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='x3d1')(out_common)
    x3 = BatchNormalization(name='x3bn1')(x3)

    if output_bias is not None:
        out_bias_output1 = tf.keras.initializers.Constant(output_bias[0])
        output_3 = Dense(1, bias_initializer=out_bias_output1, activation='relu',
                         name='output1', dtype='float32')(x3)
    else:
        output_3 = Dense(1, activation='relu', name='output1', dtype='float32')(x3)

    model = keras.Model(inputs=input_1,  # [encoder_feat.input, encoder_image.input],
                        outputs=output_3,
                        name='single-task_model')

    return model

def cnn3d_v2(input_dim, output_bias=None, kern_reg_l2=0.001, dropout=0.5):
    if output_bias is not None:
        out_bias_output1 = tf.keras.initializers.Constant(output_bias[0])

    input_1 = keras.Input(shape=input_dim, name='image')

    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1')(input_1)
    x = BatchNormalization(name='bn_1')(x)
    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_1')(x)

    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_2')(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5')(x)
    x = BatchNormalization(name='bn_5')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_6')(x)
    x = BatchNormalization(name='bn_6')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_3')(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7')(x)
    x = BatchNormalization(name='bn_7')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8')(x)
    x = BatchNormalization(name='bn_8')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4')(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7b')(x)
    x = BatchNormalization(name='bn_7b')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8b')(x)
    x = BatchNormalization(name='bn_8b')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4b')(x)

    x = Flatten(name='flt_1')(x)  # LEO MOD
    x = BatchNormalization(name='bn_9')(x)
    x = Dropout(dropout, name='dpt_1')(x)

    out_common = BatchNormalization(name='out_common')(x)

    # predict output1
    x3 = Dense(4, activation='relu',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='x3d1')(out_common)
    x3 = BatchNormalization(name='x3bn1')(x3)
    output_3 = Dense(1, bias_initializer=out_bias_output1, activation='relu',
                     name='output1', dtype='float32')(x3)

    model = keras.Model(inputs=input_1,  # [encoder_feat.input, encoder_image.input],
                        outputs=output_3,
                        name='single-task_model')

    return model

def cnn3d_v3(input_dim, output_bias=None, kern_reg_l2=0.001, dropout=0.5):
    if output_bias is not None:
        out_bias_output1 = tf.keras.initializers.Constant(output_bias[0])

    input_1 = keras.Input(shape=input_dim, name='image')

    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1')(input_1)
    x = BatchNormalization(name='bn_1')(x)
    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_1')(x)

    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_2')(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5')(x)
    x = BatchNormalization(name='bn_5')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_6')(x)
    x = BatchNormalization(name='bn_6')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_3')(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7')(x)
    x = BatchNormalization(name='bn_7')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8')(x)
    x = BatchNormalization(name='bn_8')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4')(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7b')(x)
    x = BatchNormalization(name='bn_7b')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8b')(x)
    x = BatchNormalization(name='bn_8b')(x)
    x = GlobalMaxPool3D(name='mxp_4b')(x)

    x = Flatten(name='flt_1')(x)  # LEO MOD
    x = BatchNormalization(name='bn_9')(x)
    x = Dropout(dropout, name='dpt_1')(x)

    out_common = BatchNormalization(name='out_common')(x)

    # predict output1
    x3 = Dense(4, activation='relu',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='x3d1')(out_common)
    x3 = BatchNormalization(name='x3bn1')(x3)
    output_3 = Dense(1, bias_initializer=out_bias_output1, activation='relu',
                     name='output1', dtype='float32')(x3)

    model = keras.Model(inputs=input_1,  # [encoder_feat.input, encoder_image.input],
                        outputs=output_3,
                        name='single-task_model')

    return model



def cnn3d_cole(input_dim, output_bias=None, kern_reg_l2=0.001, dropout=0.5):
    if output_bias is not None:
        out_bias_output1 = tf.keras.initializers.Constant(output_bias[0])
    else:
        out_bias_output1 = None

    padding='same'
    strides=(1,1,1)

    input_1 = keras.Input(shape=input_dim, name='image')

    x = Conv3D(8, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1a')(input_1)
    x = ReLU(name='relu_1a')(x)
    x = Conv3D(8, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1b')(x)
    x = BatchNormalization(name='bn_1a')(x)
    x = ReLU(name='relu_1b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_1a')(x)

    x = Conv3D(16, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2a')(x)
    x = ReLU(name='relu_2a')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2b')(x)
    x = BatchNormalization(name='bn_2a')(x)
    x = ReLU(name='relu_2b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_2a')(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3a')(x)
    x = ReLU(name='relu_3a')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3b')(x)
    x = BatchNormalization(name='bn_3a')(x)
    x = ReLU(name='relu_3b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_3a')(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4a')(x)
    x = ReLU(name='relu_4a')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4b')(x)
    x = BatchNormalization(name='bn_4a')(x)
    x = ReLU(name='relu_4b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_4a')(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5a')(x)
    x = ReLU(name='relu_5a')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5b')(x)
    x = BatchNormalization(name='bn_5a')(x)
    x = ReLU(name='relu_5b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_5a')(x)

    x = Flatten(name='flt_6')(x)

    x = Dropout(dropout, name='dpt_1')(x)

    out_common = BatchNormalization(name='out_common')(x)

    # predict output1
    output_1 = Dense(1, bias_initializer=out_bias_output1, activation='relu',
                     name='output1', dtype='float32')(out_common)

    model = keras.Model(inputs=input_1,
                        outputs=output_1,
                        name='single-task_model')

    return model

def cnn3d_cole_v2(input_dim, output_bias=None, kern_reg_l2=0.001, dropout=0.5):
    if output_bias is not None:
        out_bias_output1 = tf.keras.initializers.Constant(output_bias[0])
    else:
        out_bias_output1 = None

    padding='same'
    strides=(1,1,1)

    input_1 = keras.Input(shape=input_dim, name='image')

    x = Conv3D(8, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1a')(input_1)
    x = ReLU(name='relu_1a')(x)
    x = Conv3D(8, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1b')(x)
    x = BatchNormalization(name='bn_1a')(x)
    x = ReLU(name='relu_1b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_1a')(x)

    x = Conv3D(16, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2a')(x)
    x = ReLU(name='relu_2a')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2b')(x)
    x = BatchNormalization(name='bn_2a')(x)
    x = ReLU(name='relu_2b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_2a')(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3a')(x)
    x = ReLU(name='relu_3a')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3b')(x)
    x = BatchNormalization(name='bn_3a')(x)
    x = ReLU(name='relu_3b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_3a')(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4a')(x)
    x = ReLU(name='relu_4a')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4b')(x)
    x = BatchNormalization(name='bn_4a')(x)
    x = ReLU(name='relu_4b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_4a')(x)

    x = Flatten(name='flt_6')(x)

    x = Dropout(dropout, name='dpt_1')(x)

    out_common = BatchNormalization(name='out_common')(x)

    # predict output1
    output_1 = Dense(1, bias_initializer=out_bias_output1, activation='relu',
                     name='output1', dtype='float32')(out_common)

    model = keras.Model(inputs=input_1,
                        outputs=output_1,
                        name='single-task_model')

    return model

def cnn3d_cole_coord(input_dim, output_bias=None, kern_reg_l2=0.001, dropout=0.5):
    if output_bias is not None:
        out_bias_output1 = tf.keras.initializers.Constant(output_bias[0])

    padding='same'
    strides=(1,1,1)

    input_1 = keras.Input(shape=input_dim, name='image')

    x = CoordinateChannel3D()(input_1)

    x = Conv3D(8, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1a')(x)
    x = ReLU(name='relu_1a')(x)
    x = Conv3D(8, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1b')(x)
    x = BatchNormalization(name='bn_1a')(x)
    x = ReLU(name='relu_1b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_1a')(x)

    x = Conv3D(16, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2a')(x)
    x = ReLU(name='relu_2a')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2b')(x)
    x = BatchNormalization(name='bn_2a')(x)
    x = ReLU(name='relu_2b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_2a')(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3a')(x)
    x = ReLU(name='relu_3a')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3b')(x)
    x = BatchNormalization(name='bn_3a')(x)
    x = ReLU(name='relu_3b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_3a')(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4a')(x)
    x = ReLU(name='relu_4a')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4b')(x)
    x = BatchNormalization(name='bn_4a')(x)
    x = ReLU(name='relu_4b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_4a')(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5a')(x)
    x = ReLU(name='relu_5a')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), padding=padding, strides=strides,
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5b')(x)
    x = BatchNormalization(name='bn_5a')(x)
    x = ReLU(name='relu_5b')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='mxp_5a')(x)

    x = Flatten(name='flt_6')(x)

    x = Dropout(dropout, name='dpt_1')(x)

    out_common = BatchNormalization(name='out_common')(x)

    # predict output1
    output_1 = Dense(1, bias_initializer=out_bias_output1, activation='relu',
                     name='output1', dtype='float32')(out_common)

    model = keras.Model(inputs=input_1,
                        outputs=output_1,
                        name='single-task_model')

    return model

def cnn3d_class(input_dim, output_bias=None, kern_reg_l2=0.001, dropout=0.5):
    if output_bias is not None:
        out_bias_output1 = tf.keras.initializers.Constant(output_bias[0])

    input_1 = keras.Input(shape=input_dim, name='image')

    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1')(input_1)
    x = BatchNormalization(name='bn_1')(x)
    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_1')(x)

    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_2')(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5')(x)
    x = BatchNormalization(name='bn_5')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_6')(x)
    x = BatchNormalization(name='bn_6')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_3')(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7')(x)
    x = BatchNormalization(name='bn_7')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8')(x)
    x = BatchNormalization(name='bn_8')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4')(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7b')(x)
    x = BatchNormalization(name='bn_7b')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8b')(x)
    x = BatchNormalization(name='bn_8b')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4b')(x)

    x = Flatten(name='flt_1')(x)  # LEO MOD
    x = BatchNormalization(name='bn_9')(x)
    x = Dropout(dropout, name='dpt_1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(kern_reg_l2), name='d_1')(x)  # 'MtlBestTry01'
    # x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(kern_reg_l2), name='d_1')(x) # 'MtlBestTry03'

    out_common = BatchNormalization(name='out_common')(x)

    # encoder_image = keras.Model(inputs=input_1, outputs=out_common,
    #                       name='encoder_image')

    # print('Out_0:',tf.shape(out_0))
    # print('Out_common:',tf.shape(out_common))

    # predict output1
    x3 = Dense(4, activation='relu',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='x3d1')(out_common)
    x3 = BatchNormalization(name='x3bn1')(x3)
    output_3 = Dense(1, bias_initializer=out_bias_output1, activation='sigmoid',
                     name='output1', dtype='float32')(x3)

    model = keras.Model(inputs=input_1,  # [encoder_feat.input, encoder_image.input],
                        outputs=output_3,
                        name='single-task_model')

    return model

def cnn3d_coord(input_dim, output_bias=None, kern_reg_l2=0.001, dropout=0.5):
    if output_bias is not None:
        out_bias_output1 = tf.keras.initializers.Constant(output_bias[0])

    input_1 = keras.Input(shape=input_dim, name='image')

    x = CoordinateChannel3D()(input_1)
    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_1')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_1')(x)

    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_3')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_4')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_2')(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_5')(x)
    x = BatchNormalization(name='bn_5')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_6')(x)
    x = BatchNormalization(name='bn_6')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_3')(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7')(x)
    x = BatchNormalization(name='bn_7')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8')(x)
    x = BatchNormalization(name='bn_8')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4')(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_7b')(x)
    x = BatchNormalization(name='bn_7b')(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='cnv_8b')(x)
    x = BatchNormalization(name='bn_8b')(x)
    x = MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='mxp_4b')(x)

    x = Flatten(name='flt_1')(x)  # LEO MOD
    x = BatchNormalization(name='bn_9')(x)
    x = Dropout(dropout, name='dpt_1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(kern_reg_l2), name='d_1')(x)  # 'MtlBestTry01'
    # x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(kern_reg_l2), name='d_1')(x) # 'MtlBestTry03'

    out_common = BatchNormalization(name='out_common')(x)

    # encoder_image = keras.Model(inputs=input_1, outputs=out_common,
    #                       name='encoder_image')

    # print('Out_0:',tf.shape(out_0))
    # print('Out_common:',tf.shape(out_common))

    # predict output1
    x3 = Dense(4, activation='relu',
               kernel_regularizer=regularizers.l2(kern_reg_l2), name='x3d1')(out_common)
    x3 = BatchNormalization(name='x3bn1')(x3)
    output_3 = Dense(1, bias_initializer=out_bias_output1, activation='relu',
                     name='output1', dtype='float32')(x3)

    model = keras.Model(inputs=input_1,  # [encoder_feat.input, encoder_image.input],
                        outputs=output_3,
                        name='single-task_model')

    return model