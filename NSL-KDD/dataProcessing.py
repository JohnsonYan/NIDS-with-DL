#coding=utf-8

from keras.utils.data_utils import get_file
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.normalization import BatchNormalization

from helpfulTools import *


def init_data(inputfile, outputfile):
    try:
        path = get_file(inputfile, origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz')
    except:
        print('Error get file', inputfile)
        raise

    print(path)

    # This file is a CSV, just no CSV extension or headers
    # Download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    df = pd.read_csv(path, header=None)

    print("Read {} rows.".format(len(df)))
    # df = df.sample(frac=0.1, replace=False) # Uncomment this line to sample only 10% of the dataset
    #df.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)
    # The CSV file has no column heads, so add them
    df.columns = [
        'duration',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'outcome',
        'protocol_tcp',
        'protocol_udp',
        'protocol_icmp',
        'service_aol',
        'service_auth',
        'service_bgp',
        'service_courier',
        'service_csnet_ns',
        'service_ctf',
        'service_daytime',
        'service_discard',
        'service_domain',
        'service_domain_u',
        'service_echo',
        'service_eco_i',
        'service_ecr_i',
        'service_efs',
        'service_exec',
        'service_finger',
        'service_ftp',
        'service_ftp_data',
        'service_gopher',
        'service_harvest',
        'service_hostnames',
        'service_http',
        'service_http_2784',
        'service_http_443',
        'service_http_8001',
        'service_imap4',
        'service_IRC',
        'service_iso_tsap',
        'service_klogin',
        'service_kshell',
        'service_ldap',
        'service_link',
        'service_login',
        'service_mtp',
        'service_name',
        'service_netbios_dgm',
        'service_netbios_ns',
        'service_netbios_ssn',
        'service_netstat',
        'service_nnsp',
        'service_nntp',
        'service_ntp_u',
        'service_other',
        'service_pm_dump',
        'service_pop_2',
        'service_pop_3',
        'service_printer',
        'service_private',
        'service_red_i',
        'service_remote_job',
        'service_rje',
        'service_shell',
        'service_smtp',
        'service_sql_net',
        'service_ssh',
        'service_sunrpc',
        'service_supdup',
        'service_systat',
        'service_telnet',
        'service_tftp_u',
        'service_tim_i',
        'service_time',
        'service_urh_i',
        'service_urp_i',
        'service_uucp',
        'service_uucp_path',
        'service_vmnet',
        'service_whois',
        'service_X11',
        'service_Z39_50',
        'flag_OTH',
        'flag_REJ',
        'flag_RSTO',
        'flag_RSTOS0',
        'flag_RSTR',
        'flag_S0',
        'flag_S1',
        'flag_S2',
        'flag_S3',
        'flag_SF',
        'flag_SH'
    ]

    # display 5 rows
    print(df[0:5])


    # Now encode the feature vector
    encode_numeric_range(df,'duration',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'src_bytes',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_bytes',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'land',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'wrong_fragment',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'urgent',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'hot',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_failed_logins',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'logged_in',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_compromised',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'root_shell',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'su_attempted',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_root',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_file_creations',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_shells',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_access_files',normalized_low=0, normalized_high=1)
    #encode_numeric_range(df,'num_outbound_cmds',normalized_low=0, normalized_high=1)
    #encode_numeric_range(df,'is_host_login',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'is_guest_login',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'count',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'srv_count',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'serror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'srv_serror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'rerror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'srv_rerror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'same_srv_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'diff_srv_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'srv_diff_host_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_count',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_srv_count',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_same_srv_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_diff_srv_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_same_src_port_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_srv_diff_host_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_serror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_srv_serror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_rerror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_srv_rerror_rate',normalized_low=0, normalized_high=1)
    outcomes = encode_text_index(df, 'outcome')
    num_classes = len(outcomes)
    print('outcomes length:', num_classes)

    print(df[0:5])

    # save to file，不加序号
    df.to_csv(outputfile, index=False)
    print('successfully saved file:', outputfile)

def init_data_2_class(inputfile, outputfile):
    try:
        path = get_file(inputfile, origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz')
    except:
        print('Error get file', inputfile)
        raise

    print(path)

    # This file is a CSV, just no CSV extension or headers
    # Download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    df = pd.read_csv(path, header=None)

    print("Read {} rows.".format(len(df)))
    # df = df.sample(frac=0.1, replace=False) # Uncomment this line to sample only 10% of the dataset
    #df.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)
    # The CSV file has no column heads, so add them
    df.columns = [
        'duration',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'outcome',
        'protocol_tcp',
        'protocol_udp',
        'protocol_icmp',
        'service_aol',
        'service_auth',
        'service_bgp',
        'service_courier',
        'service_csnet_ns',
        'service_ctf',
        'service_daytime',
        'service_discard',
        'service_domain',
        'service_domain_u',
        'service_echo',
        'service_eco_i',
        'service_ecr_i',
        'service_efs',
        'service_exec',
        'service_finger',
        'service_ftp',
        'service_ftp_data',
        'service_gopher',
        'service_harvest',
        'service_hostnames',
        'service_http',
        'service_http_2784',
        'service_http_443',
        'service_http_8001',
        'service_imap4',
        'service_IRC',
        'service_iso_tsap',
        'service_klogin',
        'service_kshell',
        'service_ldap',
        'service_link',
        'service_login',
        'service_mtp',
        'service_name',
        'service_netbios_dgm',
        'service_netbios_ns',
        'service_netbios_ssn',
        'service_netstat',
        'service_nnsp',
        'service_nntp',
        'service_ntp_u',
        'service_other',
        'service_pm_dump',
        'service_pop_2',
        'service_pop_3',
        'service_printer',
        'service_private',
        'service_red_i',
        'service_remote_job',
        'service_rje',
        'service_shell',
        'service_smtp',
        'service_sql_net',
        'service_ssh',
        'service_sunrpc',
        'service_supdup',
        'service_systat',
        'service_telnet',
        'service_tftp_u',
        'service_tim_i',
        'service_time',
        'service_urh_i',
        'service_urp_i',
        'service_uucp',
        'service_uucp_path',
        'service_vmnet',
        'service_whois',
        'service_X11',
        'service_Z39_50',
        'flag_OTH',
        'flag_REJ',
        'flag_RSTO',
        'flag_RSTOS0',
        'flag_RSTR',
        'flag_S0',
        'flag_S1',
        'flag_S2',
        'flag_S3',
        'flag_SF',
        'flag_SH'
    ]

    # display 5 rows
    print(df[0:5])


    # Now encode the feature vector
    encode_numeric_range(df,'duration',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'src_bytes',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_bytes',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'land',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'wrong_fragment',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'urgent',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'hot',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_failed_logins',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'logged_in',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_compromised',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'root_shell',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'su_attempted',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_root',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_file_creations',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_shells',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'num_access_files',normalized_low=0, normalized_high=1)
    #encode_numeric_range(df,'num_outbound_cmds',normalized_low=0, normalized_high=1)
    #encode_numeric_range(df,'is_host_login',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'is_guest_login',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'count',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'srv_count',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'serror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'srv_serror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'rerror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'srv_rerror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'same_srv_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'diff_srv_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'srv_diff_host_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_count',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_srv_count',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_same_srv_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_diff_srv_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_same_src_port_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_srv_diff_host_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_serror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_srv_serror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_rerror_rate',normalized_low=0, normalized_high=1)
    encode_numeric_range(df,'dst_host_srv_rerror_rate',normalized_low=0, normalized_high=1)
    # outcomes = encode_text_index(df, 'outcome')
    # num_classes = len(outcomes)
    # print('outcomes length:', num_classes)

    print(df[0:5])

    # save to file，不加序号
    df.to_csv(outputfile, index=False)
    print('successfully saved file:', outputfile)

def start():
    df = pd.read_csv('./dataset/train.csv')
    # df = pd.read_csv('./dataset/train.csv')
    df_test = pd.read_csv('./dataset/test.csv')
    #df_test = pd.read_csv('./dataset/train.csv')
    # Break into X (predictors) & y (prediction)
    x, y = to_xy(df,'outcome')
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=6)
    x_test, y_test = to_xy(df_test, 'outcome')
    print('x,y:', x.shape, y.shape)
    print('x_test,y_test:', x_test.shape, y_test.shape)

    model = Sequential()
    dropout = 0.5

    # 1：
    # model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, kernel_initializer='normal'))
    # model.add(Dense(y.shape[1],activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    # model.fit(x,y,validation_data=(x_test,y_test),callbacks=[monitor],verbose=1,epochs=1000)

    # 4: 91.56927%
    # model.add(Dense(24, input_dim=x.shape[1], activation='relu', kernel_initializer='normal'))
    model.add(Dense(122, input_dim=x.shape[1], activation='relu', kernel_initializer='normal'))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(dropout))
    model.add(Dense(5, activation='softmax'))
    #rmsprop 
    #categorical_crossentropy
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0, save_best_only=True)  # 将模型保存到h5文件
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True) # 保存训练进度文件


    # 训练模型，以 32 个样本为一个 batch 进行迭代
    # monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    # model.fit(x, y, callbacks=[monitor], epochs=1000, batch_size=32)
    # epochs:5 91.5342%(corrected)
    history = model.fit(x, y, epochs=6, batch_size=32, 
                validation_data=(x_test, y_test), verbose=1,
                callbacks=[checkpointer, tensorboard]).history

    trained_model = load_model('model.h5')  # 加载保存的模型
    print("successfully load trained model: model.h5")

    # 打印模型的基本结构
    print(trained_model.summary())

    # Measure accuracy
    score = trained_model.evaluate(x_test, y_test,batch_size=32)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    # pred = model.predict(x_test)
    # pred = np.argmax(pred,axis=1)
    # y_eval = np.argmax(y_test,axis=1)
    # score = metrics.accuracy_score(y_eval, pred)
    # print("Validation score: {}".format(score))

    # return history

def start_2_class():
    df = pd.read_csv('./dataset/train_2_class.csv')
    # df = pd.read_csv('./dataset/train.csv')
    df_test = pd.read_csv('./dataset/test_2_class.csv')
    #df_test = pd.read_csv('./dataset/train.csv')
    # Break into X (predictors) & y (prediction)
    y = df.outcome
    df.drop('outcome', axis=1, inplace=True)
    x = df
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=6)
    y_test = df_test.outcome
    df_test.drop('outcome', axis=1, inplace=True)
    x_test = df_test
    print('x,y:', x.shape, y.shape)
    print('x_test,y_test:', x_test.shape, y_test.shape)

    model = Sequential()
    dropout = 0.5

    # 1：
    # model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, kernel_initializer='normal'))
    # model.add(Dense(y.shape[1],activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    # model.fit(x,y,validation_data=(x_test,y_test),callbacks=[monitor],verbose=1,epochs=1000)

    # 4: 91.56927%
    model.add(Dense(24, input_dim=x.shape[1], activation='relu', kernel_initializer='normal'))
    # model.add(Dense(122, input_dim=x.shape[1], activation='relu', kernel_initializer='normal'))
    # model.add(Dropout(dropout))
    # model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    # model.add(Dropout(dropout))
    # model.add(Dense(512, activation='relu', kernel_initializer='normal'))
    # model.add(Dropout(dropout))
    # model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    #rmsprop 
    #categorical_crossentropy
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0, save_best_only=True)  # 将模型保存到h5文件
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True) # 保存训练进度文件


    # 训练模型，以 32 个样本为一个 batch 进行迭代
    # monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    # model.fit(x, y, callbacks=[monitor], epochs=1000, batch_size=32)
    # epochs:5 91.5342%(corrected)
    history = model.fit(x, y, epochs=6, batch_size=32, 
                validation_data=(x_test, y_test), verbose=1,
                callbacks=[checkpointer, tensorboard]).history

    trained_model = load_model('model.h5')  # 加载保存的模型
    print("successfully load trained model: model.h5")

    # 打印模型的基本结构
    print(trained_model.summary())

    # Measure accuracy
    score = trained_model.evaluate(x_test, y_test,batch_size=32)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    # pred = model.predict(x_test)
    # pred = np.argmax(pred,axis=1)
    # y_eval = np.argmax(y_test,axis=1)
    # score = metrics.accuracy_score(y_eval, pred)
    # print("Validation score: {}".format(score))

    # return history


if __name__ == '__main__':
    # init_data('/Users/johnson/Downloads/graduation_project/code/NSL-KDD/dataset/train_handled.csv', './dataset/train.csv')
    # init_data('/Users/johnson/Downloads/graduation_project/code/NSL-KDD/dataset/test_handled.csv', './dataset/test.csv')
    # init_data_2_class('/Users/johnson/Downloads/graduation_project/code/NSL-KDD/dataset/train_handled_2_class.csv', './dataset/train_2_class.csv')
    # init_data_2_class('/Users/johnson/Downloads/graduation_project/code/NSL-KDD/dataset/test_handled_2_class.csv', './dataset/test_2_class.csv')
    start_2_class()
