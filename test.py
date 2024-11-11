import pandas as pd

file_path = "auto-mpg.csv"

data = pd.read_csv(file_path)

train_data = data.sample(frac=0.8, random_state=42)  # Pega 80% dos dados
test_data = data.drop(train_data.index)      

print(data)