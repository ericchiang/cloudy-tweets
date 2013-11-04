
import pickle as pk
import sys
import numpy
from sklearn.cross_validation import KFold
from sklearn.linear_model import *
from sklearn.decomposition import PCA
from feature_generation import *
from format_data import *
from utils import *

result_dir = "results"
data_dir = "data"
test_file_name = data_dir + "/test.csv"
train_file_name = data_dir + "/test.csv"
X_train_file_name = result_dir + "/X_train.pk"
X_test_file_name  = result_dir + "/X_test.pk"

var_names = ['s1','s2','s3','s4','s5',
             'w1','w2','w3','w4',
             'k1','k2','k3','k4','k5','k6','k7','k8',
             'k9','k10','k11','k12','k13','k14','k15']

def classify(var):
    X_train_file = open(X_train_file_name,'r')
    X_train = pk.load(X_train_file)
    X_train_file.close()
    

if __name__ == '__main__':
    if len(sys.argv) < 2
