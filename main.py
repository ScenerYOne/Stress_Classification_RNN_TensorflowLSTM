from function.test_gpu import check_gpu
from function.preprocess import preprocess
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
import keras_tuner as kt
from function.hypermodel import MyHyperModel
import io
from contextlib import redirect_stdout


#####Check gpu use####
check_gpu()

#####Preprocess data####
dataset = "Dataset/combined_data.csv"
dataframe = pd.read_csv(dataset)

columns = ['HRV_SDNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF', 'HRV_LFHF',
    'EDA_Phasic', 'SCR_Amplitude', 'EDA_Tonic', 'SCR_Onsets',
    'gender', 'bmi', 'sleep', 'type', 'stress', 'id']

preprocess = preprocess(dataframe, columns)
dataframe = preprocess.select_columns()
# print(dataframe.head())
# print(dataframe.tail())

print(dataframe['stress'].value_counts())

dataframe, mapping = preprocess.label_encoding()
print(dataframe.head())
print(mapping)

sc = StandardScaler()
dataframe = preprocess.scale_data(sc, ['stress', 'id'])

# print(dataframe['gender'].value_counts())
dataframe.to_csv('Dataset/combined_data_preprocessed.csv', index=False)

#######Create Sequence###############
X, y = preprocess.create_sequence()

print(X.shape)
print(y.shape)

##########Smoote Data#########

# Flatten X
X_flat = X.reshape(X.shape[0], -1)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y)

# Reshape Back
X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])

print(X_resampled.shape)
print(y_resampled.shape)

X = X_resampled
y = y_resampled

#########Split train test ##########
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X train shape : {X_train.shape}")
print(f"y train shape : {y_train.shape}")

######### Create Model and Tune Hyperparameter########

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective="val_accuracy",        # หรือเปลี่ยนเป็น metric อื่นที่อยาก optimize
    max_trials=100,                 # กี่ combinations ของ hyperparameter ที่จะลอง
    # executions_per_trial=1,           # วิ่งแต่ละครั้งกี่รอบ (สำหรับลด randomness)
    directory='my_tuner_results',
    project_name='stress_lstm',
    overwrite=True
)

tuner.search(
    X_train, y_train, # Use the sequenced data here
    X_test=X_test, y_test=y_test,
)


print(tuner.results_summary(num_trials=100))

f = io.StringIO()
with redirect_stdout(f):
    tuner.results_summary()
summary = f.getvalue()

with open('tuner_results_summary.txt', 'w') as file:
    file.write(summary)

