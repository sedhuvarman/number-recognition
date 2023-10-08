# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline


train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_df.head()



test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_df.head()


len(train_df)

len(test_df)

x_train = train_df.drop('label', axis='columns')
x_train = x_train/255
x_train.head()



y_train = train_df.label
y_train.head()



x_test = test_df
x_test = x_test/255
x_test.head()


x_train.shape

x_test.shape

model = keras.Sequential([
    # Dense means neurons of one layer is connected with every other neuron of second layer
    # here 100 is no. of hidden layers which should be less than input shape i.e. 784
    # here 10 neurons are the output (0 to 9) and input neuron is 784
    keras.layers.Dense(150, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])
# compiling neural network (optimizer allows us to train efficiently)
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)  


y_predicted = model.predict(x_test)

y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]

Label = y_predicted_labels
Label[:5]


sample_submission = pd.DataFrame({'ImageId': x_test.index+1, 'Label': Label})
sample_submission.head()