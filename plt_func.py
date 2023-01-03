import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

def scatterPlotSeaborn(df, categoryX, categoryY):
    ax = sns.scatterplot(x=df[categoryX], y=df[categoryY])
    plt.xlabel(categoryX.upper())
    plt.ylabel(categoryY.upper())
    plt.title(categoryX + " & " + categoryY, fontsize=15)

def scatterPlotMatplot(df, categoryX, categoryY):
    plt.scatter(x=df[categoryX], y=df[categoryY])
    plt.xlabel(categoryX.upper())
    plt.ylabel(categoryY.upper())
    plt.title(categoryX + " & " + categoryY, fontsize=15)

def topNScatter(top, catX, catY, name_key):
    plt.scatter(x=top[catX], y=top[catY])

    for i in range(len(top)):
        plt.text(top.iloc[i][catX], top.iloc[i][catY], top.iloc[i][name_key])
