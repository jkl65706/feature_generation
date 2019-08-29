import sys
import os
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, mean_absolute_error
from sklearn import preprocessing, neural_network
from sklearn.preprocessing import MinMaxScaler

from feature_generation.XGboost.XGboost import XGBClassifier_feature_generation
from feature_generation.XGboost.XGboost import XGBRegressor_feature_generation
from feature_generation.GBDT.GBDT import GBDTClassifier_feature_generation
from feature_generation.GBDT.GBDT import GBDTRegressor_feature_generation
from feature_generation.RF.Random_Forest import RFClassifier_feature_generation
from feature_generation.RF.Random_Forest import RFRegressor_feature_generation
from feature_generation.multiply_PCA.multiply_PCA import multiply_PCA
from feature_generation.PolynomialFeatures.Polynomial import Polynomial

def Normalize_Feature(data_path, n_feature):
    data = pd.read_csv(data_path)
    scale_feature = data.ix[:, 0:n_feature]
    #label = data.ix[:, n_feature:]
    label = data.ix[:, [n_feature]]
    data = np.concatenate((scale_feature, label.values), axis=1)
    return data

#data_path = "Data/2019_04_11_10_54_NewMCSFeaExtractM2_NN_MCS.csv"  #13 C
#data_path = "Data/pim02-0615-01_source_data.csv"                   #8  R
data_path = "Data/rank.csv"                                        #5  R
#data_path = "Data/new/pmsm_temperature_data.csv"                   #8  R
#data_path = "Data/new/dataset_6.csv"                               #12 R
#data_path = "Data/new/rank_test_data_cleanout.csv"                 #11  (3)C
#data_path = "Data/new/MbbBigPeriod_NewMCSFeaExtractM3.csv"         #13  (b)C
#data_path = "Data/new/adult.csv"                                   #14  (b)C
#data_path = "Data/new/forestfires.csv"                             #12  (b)C
#data_path = "Data/new/iris.csv"                                    #4   (b)C
#data_path = "Data/new/musk.csv"                                    #166 (b)C
classify_select = [GBDTClassifier_feature_generation, XGBClassifier_feature_generation,RFClassifier_feature_generation,
                   multiply_PCA,Polynomial]
regression_select = [GBDTRegressor_feature_generation,XGBRegressor_feature_generation, RFRegressor_feature_generation,
                     multiply_PCA, Polynomial ]
def selsect_function_by_name(name,data, n_feature, n_output):
    if name not in  classify_select or name not in  regression_select :
        print("*********not supported function*********")
    elif name == GBDTClassifier_feature_generation :
        data = GBDTClassifier_feature_generation(data, n_feature, n_output)
    elif name == XGBClassifier_feature_generation:
        data = XGBClassifier_feature_generation(data, n_feature, n_output)
    elif name == RFClassifier_feature_generation:
        data=RFClassifier_feature_generation(data, n_feature, n_output)
    elif name == GBDTRegressor_feature_generation:
        data = GBDTRegressor_feature_generation(data, n_feature, n_output)
    elif name == XGBRegressor_feature_generation:
        data = XGBRegressor_feature_generation(data, n_feature, n_output)
    elif name == RFRegressor_feature_generation:
        data = RFRegressor_feature_generation(data, n_feature, n_output)
    elif name == multiply_PCA:
        data = multiply_PCA(data, n_feature, n_output)
    elif name == Polynomial:
        data = Polynomial(data, n_feature)
    return data

if __name__ == "__main__":
    processName = psutil.Process(os.getpid())
    rss = processName.memory_info()
    print("The process base memory is:", rss.rss)
    n_feature = 5
    for n_output in range(n_feature+1, 3*n_feature, 1): #output 表示升维后的feature数
        print("*********feature generation num is*********", n_output-n_feature)
        data = Normalize_Feature(data_path, n_feature)
        #for name in classify_select:
        for name in regression_select:
            data = selsect_function_by_name(name, data, n_feature, n_output)
            if name == Polynomial:  #多项式
                n_output=3*n_feature
            # 归一化。
            scale_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
            scale_feature = scale_data[:, 0:n_output + 2 * (n_output - n_feature)]
            label = data[:, [n_output + 2 * (n_output - n_feature)]]
            data = np.concatenate((scale_feature, label), axis=1)
            x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
                data[:, 0:n_output + 2 * (n_output - n_feature)], data[:, n_output + 2 * (n_output - n_feature):],
                test_size=0.2, random_state=0)
            feature_types = (['numerical'] * (n_output + 2 * (n_output - n_feature)))
            # 回归
            MLP = neural_network.MLPRegressor((50,), learning_rate_init=0.01, max_iter=2000, )
            processName = psutil.Process(os.getpid())
            rss = processName.memory_info()
            print("The process after configure memory is:", rss.rss)
            MLP.fit(x_train_1, y_train_1)
            processName = psutil.Process(os.getpid())
            rss = processName.memory_info()
            print("The process after fit memory is:", rss.rss)
            y_pred_1 = MLP.predict(x_train_1)
            MSE_train = mean_squared_error(y_train_1, y_pred_1)
            y_pred_2 = MLP.predict(x_test_1)
            MSE_test = mean_squared_error(y_test_1, y_pred_2)
            print("MSE_train : ", MSE_train)
            print("MSE_test : ", MSE_test)

            """
            # 分类
            MLP = neural_network.MLPClassifier(hidden_layer_sizes=(4,), learning_rate_init=0.001, max_iter=2000,)
            processName = psutil.Process(os.getpid())
            rss = processName.memory_info()
            print("The process after configure memory is:", rss.rss)

            MLP.fit(x_train_1, y_train_1)

            processName = psutil.Process(os.getpid())
            rss = processName.memory_info()
            print("The process after fit memory is:", rss.rss)

            y_pred_1 = MLP.predict(x_train_1)
            MSE_train = classification_report(y_train_1, y_pred_1)
            y_pred_2 = MLP.predict(x_test_1)
            MSE_test = classification_report(y_test_1, y_pred_2)
            print("classifier_train : ", MSE_train)
            print("classifier_test : ", MSE_test)
            """

