import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import seaborn as sns  
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("vivek468/superstore-dataset-final")

print("Path to dataset files:", path)

print(os.listdir(path))

csv_path = os.path.join(path, "Sample - Superstore.csv")
df = pd.read_csv(csv_path, encoding='latin-1')

print(df.head())
print(df.info())
