import pandas as pd

train_df = pd.read_csv("artifacts/processed/engineered_train.csv")

# Display first 5 active ingredient records
for i in range(5):
    print(f"\nRecord {i+1}:")
    print(train_df["active_ingredients"].iloc[i])