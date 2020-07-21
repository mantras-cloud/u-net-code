from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans


def adjustData(img, mask, flag_multi_class, num_class):  # 将图片归一化
    if np.max(img) > 1:
        img = img / 255
        mask = mask / 255#将像素值映射到0-1范围内
        mask[mask > 0.5] = 1  # 将mask二值化
        mask[mask <= 0.5] = 0
    return img, mask


# 生成训练所需要的图片和标签
def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    # 图片生成器对数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等
    #参数就是main里面定义的字典
    image_datagen = ImageDataGenerator(**aug_dict)  
    #对image和mask分别定义图片生成器
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    #image生成器
    #flow_from_directory函数用于根据文件路径，增强数据
    image_generator = image_datagen.flow_from_directory(  
        train_path,#目标文件夹路径，从该路径提取图片来生成数据增强后的图片
        classes=[image_folder],#子文件夹列表，对于image生成器来说也就是train下的image对应的30张图片
        class_mode=None,#确定返回的标签数组的类型，none表示不返回标签数组的类型，只返回标签
        color_mode=image_color_mode,#颜色模式,grayscale为单通道，也就是灰度图像
        target_size=target_size,#目标图像的尺寸，缺省定义为256*256
        batch_size=batch_size,#表示每批数据的大小
        save_to_dir=save_to_dir,#表示保存图片，缺省值为none，也就是不保存数据增强后的图片
        save_prefix=image_save_prefix,#保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
        seed=seed)#表示随机数种子，缺省值为true，即表示每次数据增强都要打乱数据


    #mask生成器（label数据增强的结果）
    #参数与image生成器的参数一致，只需要把mask对应的值赋值给这些参数即可。
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    
    # 将image和mask打包成元组的列表[(image1,mask1),(image2,mask2),...]
    train_generator = zip(image_generator, mask_generator)  
    for (img, mask) in train_generator:
        #数据调整函数，adjustData函数在data.py文件中
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)

#测试集数据生成
def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        #读取test原图
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255  # 归一化
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)  # (1,width,height)
        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
