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

# Exibir as primeiras linhas do DataFrame e informações sobre os dados
print(df.head())
print(df.info())
print(df.describe())


# Não tem valores nulos
print(df.isnull().sum())
# Não tem valores duplicados
print(df.duplicated().sum())

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(' ', '_')
)

print(df.columns)

df['postal_code'] = df['postal_code'].astype(str)

print(df.dtypes)

sns.boxplot(x=df['sales'])
plt.show()
#Foram identificados outliers na variável VENDAS, representando vendas significativamente maiores. 
#Esses valores podem indicar pedidos excepcionais e foram mantidos para preservar a realidade do negócio.
# Caso queira remover os outliers, pode usar o método do IQR:
# Q1 = df['sales'].quantile(0.25)
#Q3 = df['sales'].quantile(0.75)

#IQR = Q3 - Q1

#limite_inferior = Q1 - 1.5 * IQR
#limite_superior = Q3 + 1.5 * IQR

#df_sem_outliers = df[
#    (df['sales'] >= limite_inferior) &
#    (df['sales'] <= limite_superior)
#] 


#ANALISE EXPLORATORIA DOS DADOS

sales_category = (
    df.groupby('category')['sales']
    .sum()
    .sort_values(ascending=False)
)

print(sales_category)

profit_category = (
    df.groupby('category')['profit']
    .sum()
    .sort_values(ascending=False)
)

print(profit_category)

top_products = (
    df.groupby('product_name')['sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products)

sales_region = (
    df.groupby('region')['sales']
    .sum()
    .sort_values(ascending=False)
)

print(sales_region)

negative_profit = df[df['profit'] < 0]

print(negative_profit.head())

print(df[['sales', 'profit', 'discount']].describe())


#GRÁFICOS

sales_category.plot(kind='bar')

plt.title('Total de Vendas por Categoria')
plt.xlabel('Categoria')
plt.ylabel('Vendas')
plt.show()

profit_region = df.groupby('region')['profit'].sum()

profit_region.plot(kind='bar')

plt.title('Lucro por Região')
plt.xlabel('Região')
plt.ylabel('Lucro')
plt.show()

sns.histplot(df['sales'], bins=30)

plt.title('Distribuição das Vendas')
plt.xlabel('Sales')
plt.show()

corr = df[['sales', 'profit', 'discount', 'quantity']].corr()

sns.heatmap(corr, annot=True)

plt.title('Correlação entre Variáveis')
plt.show()

#DESCONTO X LUCRO

sns.scatterplot(
    data=df,
    x='discount',
    y='profit'
)

plt.title('Desconto vs Lucro')
plt.xlabel('Discount')
plt.ylabel('Profit')
plt.show()


