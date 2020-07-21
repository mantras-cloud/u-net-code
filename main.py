from model import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#定义了一个字典，参数是数据增强的相关参数
data_gen_args = dict(rotation_range=0.2,  # 旋转
                     width_shift_range=0.05,  # 宽度变化
                     height_shift_range=0.05,  # 高度变化
                     shear_range=0.05,  # 错切变换
                     zoom_range=0.05,  # 缩放
                     horizontal_flip=True,  # 水平翻转
                     fill_mode='nearest')  # 填充模式
#生成训练所需要的图片和标签，trainGenerator函数在data.py
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)

model_use = unet()#初始化unet模型
#保存模型
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
# 在每个epoch后保存模型到filepath
# filename:保存模型的路径
# mointor:需要监视的值
# verbose:表示信息展示模式，1展示，0不展示
# save_best_only:保存在验证集上性能最好的模型


model_use.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
# 训练函数
# generator:生成器(image,mask)
# steps_per_epoch:训练steps_per_epoch个数据时记一个epoch结束
# epoch:数据迭代轮数
# callbacks:回调函数

#测试集数据增强生成，testGenerator位于data.py文件
testGene = testGenerator("data/membrane/test")
# 为测试图片生成预测
results = model_use.predict_generator(testGene, 30, verbose=1)
# generator:生成器
# steps:在声明一个epoch完成,并开始下一个epoch之前从生成器产生的总步数
# verbose:信息展示模式，1展示，0不展示

#保存测试结果
saveResult("data/membrane/test", results)
