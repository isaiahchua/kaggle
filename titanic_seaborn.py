import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context('paper', font_scale=1.4)
sns.set_style('dark')

# import dataset
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')

# using seaborn
df = dataset.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
df_test = testset.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
fig, axs = plt.subplots(2, 3, figsize=(12, 10), dpi=300)
fig.set_tight_layout(True)
sns.barplot(ax=axs[0, 0], x='Sex', y='Survived', data=df)
sns.barplot(ax=axs[0, 1], x='Pclass', y='Survived', data=df)
sns.barplot(ax=axs[0, 2], x=df['SibSp'] + df['Parch'], y='Survived', data=df)
sns.barplot(ax=axs[1, 0], x='Embarked', y='Survived', data=df)
sns.histplot(ax=axs[1, 1], x=df[df['Survived']==0].loc[:, 'Age'], data=df, bins=10, stat='probability')
sns.histplot(ax=axs[1, 2], x=df[df['Survived']==1].loc[:, 'Age'], data=df, bins=10, stat='probability')
df_fg = sns.FacetGrid(df, row='Pclass', col='Embarked')
df_fg.map(plt.hist, 'Survived', **dict(histtype='bar', width=0.25))
for ax in axs.flatten()[:4]:
    ax.set_ylim((0, 1))
for ax in axs.flatten()[4:]:
    ax.xaxis.set_ticks(np.linspace(0, 80, 9))
axs[0, 2].set_xlabel('Family_Onboard')
axs[1, 1].set_ylabel('Density (Deceased)')
axs[1, 2].set_ylabel('Density (Alive)')
df_fg.add_legend()
fig.savefig('sns_plot.png')
