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

def delete(inputfile, outputfile):
    try:
        path = get_file(inputfile, origin='http://localhost/kddcup.data.gz')
    except:
        print('Error get file', inputfile)
        raise

    print(path)

    # This file is a CSV, just no CSV extension or headers
    df = pd.read_csv(path, header=None)
    print("Read {} rows.".format(len(df)))

    df.columns = [
        'duration',
        'protocol_type',
        'service',
        'flag',
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
        'type_no'
    ]

    df.drop('type_no', axis=1, inplace=True)
    
    df.to_csv(outputfile, header=False,index=False)
    print('successfully saved file:', outputfile)

if __name__ == '__main__':
    delete('/Users/johnson/Downloads/graduation_project/code/NSL-KDD/dataset/KDDTrain+.txt', './dataset/train-unhandled.csv')
    delete('/Users/johnson/Downloads/graduation_project/code/NSL-KDD/dataset/KDDTest+.txt', './dataset/test-unhandled.csv')
