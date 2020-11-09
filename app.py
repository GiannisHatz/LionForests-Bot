from flask import Flask, render_template, request, jsonify
from LionForests.LionForests import LionForests
from LFBot import LFBot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import urllib
import warnings

warnings.filterwarnings("ignore")
'''
# Heart Statlog Dataset Setup Below.
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
raw_data = urllib.request.urlopen(url)
credit = np.genfromtxt(raw_data)
X, y = credit[:, :5], credit[:, -1].squeeze()
print(X)
feature_names = ['age', 'sex', 'chest pain', 'resting blood pressure', 'serum cholestoral']
class_names = ['absence', 'presence']
y = [int(i - 1) for i in y]
parameters = [{
    'max_depth': [5],
    'max_features': ['sqrt'],
    'bootstrap': [False],
    'min_samples_leaf': [5],
    'n_estimators': [500]
}]
discrete_features = ['age']
categorical_features = ['sex', 'chest pain']
categorical_map = {
    'sex': ['Female', 'Male'],
    'chest pain': ['dummy', 'Asymptomatic', 'Non-anginal pain', 'Atypical angina', 'Typical angina']
}
lf = LionForests(class_names=class_names)
scaler = MinMaxScaler(feature_range=(-1, 1))
lf.train(X, y, scaler, feature_names, parameters)
'''
# Banknote Dataset Setup Below

banknote_datadset = pd.read_csv('https://raw.githubusercontent.com/Kuntal-G/Machine-Learning/master/R-machine-learning/data/banknote-authentication.csv')
feature_names = ['variance','skew','curtosis','entropy']
class_names=['fake banknote','real banknote'] #0: no, 1: yes #or ['not authenticated banknote','authenticated banknote']
X = banknote_datadset.iloc[:, 0:4].values
y = banknote_datadset.iloc[:, 4].values
parameters = [{
    'max_depth': [10],
    'max_features': [0.75],
    'bootstrap': [True],
    'min_samples_leaf' : [1],
    'n_estimators': [500]
}]
lf = LionForests(class_names=class_names)
scaler = MinMaxScaler(feature_range=(-1,1))
lf.train(X, y, scaler, feature_names, parameters)
categorical_features = []
discrete_features = []
categorical_map = []

# Adult Census Dataset Setup Below
'''
feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                 'salary']
class_names = ['<=50K', '>50K']  # 0: <=50K and 1: >50K
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=feature_names,
                   delimiter=', ')
data_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                        names=feature_names, delimiter=', ')
data_test = data_test.drop(data_test.index[[0]])
data = data[(data != '?').all(axis=1)]
data_test = data_test[(data_test != '?').all(axis=1)]
data_test['salary'] = data_test['salary'].map({'<=50K.': '<=50K', '>50K.': '>50K'})
frames = [data, data_test]
data = pd.concat(frames)
data['salary'] = data['salary'].map({'<=50K': 0, '>50K': 1})
data['sex'] = data['sex'].map({'Male': 2, 'Female': 1})
data['age'] = list(map(int, data['age']))
data = data.drop(data.index[data['relationship'] == 'Other-relative'])
data = data.drop(data.index[data['relationship'] == 'Own-child'])
data['relationship'] = data['relationship'].map({'Husband': 4, 'Wife': 3, 'Unmarried': 2, 'Not-in-family': 1})
data = data.drop(data.index[data['education'] == 'Preschool'])
data = data.drop(data.index[data['education'] == 'Assoc-voc'])
data = data.drop(data.index[data['education'] == 'Assoc-acdm'])
data = data.drop(data.index[data['education'] == '9th'])
data = data.drop(data.index[data['education'] == '7th-8th'])
data = data.drop(data.index[data['education'] == '1st-4th'])
data = data.drop(data.index[data['education'] == '12th'])
data = data.drop(data.index[data['education'] == '11th'])
data = data.drop(data.index[data['education'] == '10th'])
data = data.drop(data.index[data['education'] == '5th-6th'])
data['education'] = data['education'].map({'HS-grad': 1, 'Some-college': 2, 'Bachelors': 3,
                                           'Masters': 4, 'Prof-school': 5, 'Doctorate': 6})
data = data.drop(['workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation', 'race', 'capital-gain',
                  'capital-loss', 'native-country'], axis=1)
feature_names = ['age', 'education', 'relationship', 'sex', 'hours-per-week']
class_names = ['<=50K', '>50K']  # 0: <=50K and 1: >50K
X = data.loc[:, data.columns != "salary"].values
y = data['salary'].values
parameters = [{
    'max_depth': [10],
    'max_features': ['sqrt'],
    'bootstrap': [False],
    'min_samples_leaf': [1],
    'n_estimators': [100]
}]
lf = LionForests(class_names=class_names)
categorical_map = {
    'education': ['dummy', 'HS-grad', 'Some-College', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'],
    'relationship': ['dummy', 'Unmarried', 'Not-in-family', 'Wife', 'Husband'],
    'sex': ['dummy', 'Female', 'Male']
}
categorical_features = ['sex', 'relationship', 'education']
discrete_features = ['age', 'hours-per-week']
lf.train(X, y, None, feature_names, parameters)
'''
if __name__ == "__main__":
    #description = "Hey there! Let’s predict the absence or presence of a serious killer disease in your heart. Are you in?"  # Statlog
    description = "Hey there! Let's predict if those banknotes you've collected are valid or not. Are you in?"  # Banknote
    #description = "Hey there! Wanna live the American Dream? Let's predict if you would make over 50K per year in the US. Are you in?"  # AC
    lfbot = LFBot(X, y, feature_names, categorical_features, class_names, parameters, description, lf,
                  discrete_features, categorical_map)
    lfbot.run()
