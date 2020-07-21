import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)  # 初始化keras张量
    
    #第一层卷积
    #实际上从unet的结构来看每一次卷积的padding应该是valid，也就是每次卷积后图片尺寸减少2，
    #但在这里为了避免裁剪，方便拼接，把padding设成了same，即每次卷积不会改变图片的尺寸。
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # filters:输出的维度
    # kernel_size:卷积核的尺寸
    # activation:激活函数
    # padding:边缘填充，实际上在该实验中并没有严格按照unet网络结构进行卷积，same填充在卷积完毕之后图片大小并不会改变
    # kernel_initializer:kernel权值初始化
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#采用2*2的最大池化
    
    #第二层卷积
    #参数类似于第一层卷积，只是输出的通道数翻倍
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    #第三层卷积
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    #第四层卷积
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)  # 每次训练时随机忽略50%的神经元，减少过拟合
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    #第五层卷积
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)# 每次训练时随机忽略50%的神经元，减少过拟合
    
    #第一次反卷积
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))  # 先上采样放大，在进行卷积操作，相当于转置卷积
    # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    #将第四层卷积完毕并进行Dropout操作后的结果drop4与反卷积后的up6进行拼接
    merge6 = concatenate([drop4, up6], axis=3)  # （width,heigth,channels）拼接通道数
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    #第二次反卷积
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    # merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    #将第三层卷积完毕后的结果conv3与反卷积后的up7进行拼接
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    #第三次反卷积
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    # merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    #将第二层卷积完毕后的结果conv2与反卷积后的up8进行拼接
    merge8 = concatenate([conv2, up8], axis=3)#拼接通道数
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    
    #第四次反卷积
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    # merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    #将第一层卷积完毕后的结果conv1与反卷积后的up9进行拼接
    merge9 = concatenate([conv1, up9], axis=3)#拼接通道数
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    #进行一次卷积核为1*1的卷积操作，卷积完毕后通道数变为1，作为输出结果
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    #keras内置函数，对模型进行编译
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # optimizer:优化器
    # binary_crossentropy:与sigmoid相对应的损失函数
    # metrics:评估模型在训练和测试时的性能的指标

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
