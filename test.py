import pandas as pd
import numpy as np
import tensorflow as tf

file_path = "auto-mpg.csv"

data = pd.read_csv(file_path)

train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)      

X_train = train_data.iloc[:, 1:-1]
print(X_train)
y_train = train_data.iloc[:, 0]

X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, 0]

X_train['horsepower'] = pd.to_numeric(X_train['horsepower'])
x = np.array(X_train)
print(x)


normalizer = tf.keras.layers.Normalization()
normalizer.adapt(np.array(X_train))
data_normalized = normalizer(np.array(X_train))

print(data_normalized)