# 导入相关库
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from dataProcessing import start

%matplotlib osx

sns.set(style='whitegrid', palette='muted', font_scale=1.5)  # 配置seaborn库参数

rcParams['figure.figsize'] = 14, 8  # 设置画布大小

RANDOM_SEED = 42  # 定义随机种子
LABELS = ["NORMAL", "PROBE", "DOS", "U2R", "R2L"]  # 定义标签，全局变量使用大写

# 加载数据
df = pd.read_csv("./dataset/10_percent.csv")  # 读取数据集

# 绘制

"""
下面的代码产生了 训练集标签分布 的图表
"""
count_classes = pd.value_counts(df['outcome'], sort = False)  # 计算分类数量并排序
count_classes.plot(kind = 'bar', rot=0)  # 绘制柱状图
plt.title("Attack types distribution")  # 添加标题
plt.xticks(range(5), LABELS)  # 设置横坐标轴标签
plt.xlabel("Attack types")  # 设置x轴标题
plt.ylabel("Frequency")  # 设置y轴标题
# 为每个条形图添加数值标签
for x,y in enumerate(count_classes):
    plt.text(x, y+100,'%s' %y,ha='center')


history = start()

# step1
plt.plot(history['loss'])  # 绘制训练集loss曲线
plt.plot(history['val_loss'])  # 绘制测试集loss曲线
plt.title('model loss')  # 添加标题
plt.ylabel('loss')  # 添加y轴标题
plt.xlabel('epoch') # 添加x轴标题
plt.legend(['train', 'test'], loc='upper right')  # 添加图例

# step2
fig = plt.figure()  # 创建画布
ax = fig.add_subplot(111)  # 添加图表

predictions = autoencoder.predict(X_test)  # 使用自动编码器训练测试数据集

mse = np.mean(np.power(X_test - predictions, 2), axis=1)  # 计算重构误差值
error_df = pd.DataFrame({'reconstruction_error': mse,
                         'true_class': y_test})  # 将重构误差和真实分类放在一起

error_df.describe()  # 查看描述统计结果

# 将真实分类为0且重构误差小于10的数据作为正常误差数据
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)  # 绘制正常误差数据重构误差值直方图分布




