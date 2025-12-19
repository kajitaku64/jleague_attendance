import pandas as pd

train = pd.read_csv("data/raw/train_add.csv")
test = pd.read_csv("data/raw/test.csv")

print("=== train columns ===")
print(train.columns)

print("\n=== test columns ===")
print(test.columns)

print("\n=== train head ===")
print(train.head())
