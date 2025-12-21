import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


link="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df=pd.read_csv(link)
print(df.describe())
print(df.info())


#Filling the datas
df["bill_length_mm"] = df["bill_length_mm"].fillna(df["bill_length_mm"].mean())
df["bill_depth_mm"] = df["bill_depth_mm"].fillna(df["bill_depth_mm"].mean())





#Printing special print
selected_gender=df["sex"]=='Male'
print("It is male\n",selected_gender.tail())
selecter_tip=df["bill_length_mm"]>=40
print("The tips that are more than 2 \n",selecter_tip.head())




#Making a graphs
male_counts = df[df["sex"] == "MALE"].groupby("species").size()
total_counts = df.groupby("species").size()
male_percentage = (male_counts / total_counts) * 100
male_percentage.plot(kind="box")
plt.show()





#Dropping duplicates
df=df.drop_duplicates()








