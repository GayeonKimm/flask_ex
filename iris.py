# iris.data 파일의 학습데이터에 대해 iris 종류를 예측하는
# 머신러닝 모델 코드가 포함됨

import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.data')

X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# SVC로 모델 학습
from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)


# pickle 을 사용해 모델을 저장
pickle.dump(sv, open('iri.pkl', 'wb'))
