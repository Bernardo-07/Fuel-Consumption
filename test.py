import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def norm(input_data):
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(np.array(input_data))
    data_normalized = normalizer(np.array(input_data))
    return data_normalized

file_path = "auto-mpg.csv"

data = pd.read_csv(file_path)

for index, value in enumerate(data["horsepower"]):
        if value == "?":  # Verifica se o valor é "?"
            data.at[index, "horsepower"] = None  # Substitui por None (convertido para NaN)
            data = data.dropna()
data["horsepower"] = pd.to_numeric(data["horsepower"], errors='coerce')    

#using pandas to split the dataset
train_data = data.sample(frac=0.7, random_state=42)  # 70% para treino
remaining_data = data.drop(train_data.index)  # Remove os dados de treino

valid_data = remaining_data.sample(frac=0.5, random_state=42)  # 50% do restante para validação (15% do total)
test_data = remaining_data.drop(valid_data.index)  # O restante para teste (15% do total)

# Separando features (X) e labels (y)
X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, 0]

X_valid = valid_data.iloc[:, 1:-1]
y_valid = valid_data.iloc[:, 0]

X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, 0]

X_train = norm(X_train)
X_valid = norm(X_valid)
X_test = norm(X_test)

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='linear'))

#configuring the model for training
model.compile(
    optimizer='adam',
    loss='mse',
    loss_weights=None,
    metrics=['r2_score'],
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile='auto',
    auto_scale_loss=True
)

#training the model
modelo = model.fit(
    X_train, 
    y_train, 
    validation_data=(X_valid, y_valid),
    batch_size=32, 
    epochs=200
)


plt.plot(modelo.history['loss'], label='train')
plt.plot(modelo.history['val_loss'], label='valid')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss in Training and Validation')
plt.show()

from tensorflow.keras.models import load_model
import os

model.save(os.path.join('models', 'mpg.h5'))
new_model = load_model(os.path.join('models', 'mpg.h5'))

results = new_model.predict(X_test, y_test)
