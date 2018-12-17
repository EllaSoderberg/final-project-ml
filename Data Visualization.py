import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/Movies_data_Merged_12-14.csv', sep=';', thousands=',')

data['Box_Office'] = pd.to_numeric(data['Box_Office'], errors='coerce')

#Director vs year vs box office


data_copy = data.copy().dropna()
director_box = data_copy.groupby(data_copy['directorsNames'])['Box_Office'].sum()
director_box_index = director_box.sort_values(ascending=False)[:20].index
director_box_pivot = pd.pivot_table(data = data_copy[data_copy['directorsNames'].isin(director_box_index)],index=['Year'], columns = ['directorsNames'], values= ['Box_Office'], aggfunc = 'sum')


fig, ax = plt.subplots()
sns.heatmap(director_box_pivot['Box_Office'],vmin = 0, annot= False, linewidth=.5, ax=ax)
plt.title('Director vs Year vs Box Office')
plt.ylabel('Year')

plt.show()

#Writer vs year vs Box Office

writers_box = data_copy.groupby(data_copy['writers'])['Box_Office'].sum()
writers_box_index = writers_box.sort_values(ascending=False)[:20].index
writers_box_pivot = pd.pivot_table(data = data_copy[data_copy['writers'].isin(writers_box_index)],index=['Year'], columns = ['writers'], values= ['Box_Office'], aggfunc = 'sum')


fig, ax = plt.subplots()
sns.heatmap(writers_box_pivot['Box_Office'],vmin = 0, annot= False, linewidth=.5, ax=ax)
plt.title('Writers vs Year vs Box Office')
plt.ylabel('Year')

plt.show()

