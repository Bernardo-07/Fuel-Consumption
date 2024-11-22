import pandas as pd
import numpy as np
import tensorflow as tf

def normalizer(data):
    norm = tf.keras.layers.Normalization()
    norm.adapt(data)
    data_normalized = norm(data)
    return data_normalized

file_path = "auto-mpg.csv"

data = pd.read_csv(file_path)

for index, value in enumerate(data["horsepower"]):
        if value == "?":  # Verifica se o valor é "?"
            data.at[index, "horsepower"] = None  # Substitui por None (convertido para NaN)
            data = data.dropna()
data["horsepower"] = pd.to_numeric(data["horsepower"], errors='coerce')

train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)      

X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, 0]

X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, 0]

X = normalizer(np.array(X_train))

print(X)