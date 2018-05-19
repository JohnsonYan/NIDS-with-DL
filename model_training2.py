#coding=utf-8
import io
from keras import regularizers
from keras.utils import plot_model, to_categorical
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from keras.optimizers import SGD,Nadam
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

from helpfulTools import *


class ModelTraining(object):
    def __init__(self):
        self.df = pd.read_csv('./dataset/10_percent.csv')
        self.df_test = pd.read_csv('./dataset/test.csv')

    def start(self):
        # Break into X (predictors) & y (prediction)
        x, y = to_xy(self.df,'outcome')
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=6)
        x_test, y_test = to_xy(self.df_test, 'outcome')
        print('shape of x,y:', x.shape, y.shape)
        print('shape of x_test,y_test:', x_test.shape, y_test.shape)

        # 构建模型
        model = Sequential()
        DROPOUT = 0.5


        # model.add(Dense(24, activation='relu', input_dim=x.shape[1]))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(122, input_dim=x.shape[1], activation='relu', kernel_initializer='normal'))
        model.add(Dense(12, input_dim=x.shape[1], activation='relu', kernel_initializer='normal'))
        model.add(Dropout(DROPOUT))
        # model.add(BatchNormalization())
        model.add(Dense(128, activation='relu', kernel_initializer='normal'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(256, activation='relu', kernel_initializer='normal'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(128, activation='relu', kernel_initializer='normal'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(5, activation='softmax'))
    
        """
        optimizer:
            sgd: 80+%
            nadam:90+%
        层数目前影响不大
        dropout需要调整
        l2正则

        """
        # 编译模型
        epochs_nb = 20
        learning_rate = 0.5
        decay_rate = learning_rate / epochs_nb
        momentum = 0.8
        SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
        Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(optimizer='nadam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # define some callback functions
        checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0, save_best_only=True)  # 将模型保存到h5文件
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True) # 保存训练进度文件
        monitor = EarlyStopping(monitor='val_acc', min_delta=1e-3, patience=5, verbose=1, mode='auto')
        
        # 训练模型，以 32 个样本为一个 batch 进行迭代
        # history = model.fit(x, y, epochs=epochs_nb, batch_size=512, 
        #             validation_split=0.25, verbose=1,
        #             callbacks=[checkpointer, tensorboard, monitor]).history
        history = model.fit(x, y, epochs=epochs_nb, batch_size=512, 
                    validation_data=(x_test,y_test), verbose=1,
                    callbacks=[checkpointer, tensorboard, monitor]).history

        # 生成模型的结构图
        plot_model(model, to_file='model.png')
        # 输出训练过程中训练集和验证集准确值和损失值得变化
        print(history.keys())
        loss_values = history['loss']
        val_loss_values = history['val_loss']

        epochs = range(1, len(history['acc']) + 1)

        plt.plot(epochs, loss_values, 'bo', label='Training loss')         
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')   
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        # summarize history for loss
        plt.clf()  # 清除图片                                  
        acc_values = history['acc']
        val_acc_values = history['val_acc']

        plt.plot(epochs, acc_values, 'bo', label='Training acc')
        plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        # 加载保存的模型
        trained_model = load_model('model.h5')  
        print("successfully load trained model: model.h5")

        # 打印模型的基本结构
        print(trained_model.summary())

        # Measure accuracy
        score = trained_model.evaluate(x_test, y_test,batch_size=512)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        return history

    def start_2_class(self):
        df = pd.read_csv('./dataset/train_2_class.csv')
        df_test = pd.read_csv('./dataset/test_2_class.csv')
        # Break into X (predictors) & y (prediction)
        y = df.outcome
        df.drop('outcome', axis=1, inplace=True)
        x = df

        y_test = df_test.outcome
        df_test.drop('outcome', axis=1, inplace=True)
        x_test = df_test

        print('x,y:', x.shape, y.shape)
        print('x_test,y_test:', x_test.shape, y_test.shape)

        model = Sequential()
        DROPOUT = 0.5

        # 2-class 92.334%
        # model.add(Dense(122, input_dim=x.shape[1], activation='relu', kernel_initializer='normal'))
        # model.add(Dense(24, input_dim=x.shape[1], activation='relu', kernel_initializer='normal',kernel_regularizer=regularizers.l2(0.01)))
        # model.add(Dropout(DROPOUT))
        # model.add(Dense(256, activation='relu', kernel_initializer='normal'))
        # model.add(Dropout(DROPOUT))
        # model.add(Dense(64, activation='relu', kernel_initializer='normal'))
        # model.add(Dropout(DROPOUT))
        # model.add(Dense(8, activation='relu', kernel_initializer='normal'))
        # model.add(Dropout(DROPOUT))
        # model.add(Dense(1, activation='sigmoid'))

        # testing model structure
        model.add(Dense(16, activation='relu', input_dim=x.shape[1]))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath="model_2_class.h5", verbose=0, save_best_only=True)  # 将模型保存到h5文件
        tensorboard = TensorBoard(log_dir='./logs_2_class', histogram_freq=0, write_graph=True, write_images=True) # 保存训练进度文件

        # 训练模型，以 32 个样本为一个 batch 进行迭代
        monitor = EarlyStopping(monitor='val_acc', min_delta=1e-3, patience=3, verbose=1, mode='auto')
        
        # history = model.fit(x, y, epochs=20, batch_size=512, 
        #             validation_data=(x_test, y_test), verbose=1,
        #             callbacks=[checkpointer, tensorboard, monitor]).history
        history = model.fit(x, y, epochs=20, batch_size=512, 
                    validation_split=0.25, verbose=1,
                    callbacks=[checkpointer, tensorboard, monitor]).history

        # 生成模型的结构图
        plot_model(model, to_file='model.png')
        # 输出训练过程中训练集和验证集准确值和损失值得变化
        print(history.keys())
        loss_values = history['loss']
        val_loss_values = history['val_loss']

        epochs = range(1, len(history['acc']) + 1)

        plt.plot(epochs, loss_values, 'bo', label='Training loss')         
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')   
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        """plt.plot()
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()"""
        # summarize history for loss
        plt.clf()  # 清除图片                                  
        acc_values = history['acc']
        val_acc_values = history['val_acc']

        plt.plot(epochs, acc_values, 'bo', label='Training acc')
        plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        """plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()"""

        trained_model = load_model('model_2_class.h5')  # 加载保存的模型
        print("successfully load trained model: model_2_class.h5")

        # 打印模型的基本结构
        print(trained_model.summary())

        # Measure accuracy
        score = trained_model.evaluate(x_test, y_test,batch_size=512)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def split_xy(self, df, target):
        y = df[target]
        x = df.drop([target], axis=1)
        return x,y

    def assessment(self):
        # 加载保存的模型
        trained_model = load_model('model.h5')  
        print("successfully load trained model: model.h5")

        # 加载测试数据
        test_0 = pd.read_csv('./dataset/0_test.csv')
        test_1 = pd.read_csv('./dataset/1_test.csv')
        test_2 = pd.read_csv('./dataset/2_test.csv')
        test_3 = pd.read_csv('./dataset/3_test.csv')
        test_4 = pd.read_csv('./dataset/4_test.csv')

        # 分割数据为输入数据和标签数据
        x_test, y_test = self.split_xy(self.df_test, 'outcome')
        x_test_0, y_test_0 = self.split_xy(test_0, 'outcome')
        x_test_1, y_test_1 = self.split_xy(test_1, 'outcome')
        x_test_2, y_test_2 = self.split_xy(test_2, 'outcome')
        x_test_3, y_test_3 = self.split_xy(test_3, 'outcome')
        x_test_4, y_test_4 = self.split_xy(test_4, 'outcome')
        print(y_test_0[0])
        print(y_test_1[0])
        print(y_test_2[0])
        print(y_test_3[0])
        print(y_test_4[0])

        # 把标签数据处理成one-hot编码
        y_test = to_categorical(y_test, num_classes=5)
        y_test_0 = to_categorical(y_test_0, num_classes=5)
        y_test_1 = to_categorical(y_test_1, num_classes=5)
        y_test_2 = to_categorical(y_test_2, num_classes=5)
        y_test_3 = to_categorical(y_test_3, num_classes=5)
        y_test_4 = to_categorical(y_test_4, num_classes=5)

        print(y_test_0[0])
        print(y_test_1[0])
        print(y_test_2[0])
        print(y_test_3[0])
        print(y_test_4[0])
        print('shape of x_test,y_test:', x_test.shape, y_test.shape)
        print('shape of x_test_0,y_test_0:', x_test_0.shape, y_test_0.shape)
        print('shape of x_test_1,y_test_1:', x_test_1.shape, y_test_1.shape)
        print('shape of x_test_2,y_test_2:', x_test_2.shape, y_test_2.shape)
        print('shape of x_test_3,y_test_3:', x_test_3.shape, y_test_3.shape)
        print('shape of x_test_4,y_test_4:', x_test_4.shape, y_test_4.shape)

        # 打印模型的基本结构
        print(trained_model.summary())

        # Measure accuracy
        score = trained_model.evaluate(x_test, y_test)
        print('Total Test score:', score[0])
        print('Total Test accuracy:', score[1])
        # 评估对不同分类的准确度
        score = trained_model.evaluate(x_test_0, y_test_0)
        print('Test_0 score:', score[0])
        print('Test_0 accuracy:', score[1])

        score = trained_model.evaluate(x_test_1, y_test_1)
        print('Test_1 score:', score[0])
        print('Test_1 accuracy:', score[1])

        score = trained_model.evaluate(x_test_2, y_test_2)
        print('Test_2 score:', score[0])
        print('Test_2 accuracy:', score[1])

        score = trained_model.evaluate(x_test_3, y_test_3)
        print('Test_3 score:', score[0])
        print('Test_3 accuracy:', score[1])

        score = trained_model.evaluate(x_test_4, y_test_4)
        print('Test_4 score:', score[0])
        print('Test_4 accuracy:', score[1])


if __name__ == '__main__':
    model = ModelTraining()
    # init_data('/Users/johnson/Downloads/graduation_project/code/dataset/kddcup.data_handled.csv', 'train.csv')
    # init_data('/Users/johnson/Downloads/graduation_project/code/dataset/corrected_handled.csv', 'test.csv')
    model.start()
    # model.start_2_class()
    # model.assessment()