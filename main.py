import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math
from collections import Counter
import plt_func as pf

file = 'dataset/players_22.csv'
df = pd.read_csv(file)
print(df.head(5))

print(df.columns)

print(df.select_dtypes(include='number'))

print(df.shape)

print(df.dtypes)

print(df.describe())

#Potential & Wage
plt.figure(figsize=(7, 5))
pf.scatterPlotSeaborn(df, 'potential', 'wage_eur')
plt.show()

#Potential & Value
fig, ax = plt.subplots(figsize=(7, 5))
pf.scatterPlotMatplot(df, 'potential', 'value_eur')
plt.show()

#Reputation & Value
fig, ax = plt.subplots(figsize=(7, 5))
pf.scatterPlotMatplot(df, 'international_reputation', 'value_eur')
plt.show()

#Reputation & Wages
fig, ax = plt.subplots(figsize=(7, 5))
pf.scatterPlotMatplot(df, 'international_reputation', 'wage_eur')
plt.show()

#Height Vs Potential
fig, ax = plt.subplots(figsize=(8, 5))
pf.scatterPlotMatplot(df, 'height_cm', 'potential')
plt.show()

#Listing top 15 based on overall points
top_15 = df.nlargest(15, 'overall')
print(top_15.head(5))

#Overall Rating vs Mentality Composure of top 15
fig, ax = plt.subplots(figsize=(8, 5))

pf.topNScatter(top_15,'overall', 'mentality_composure', 'short_name')

ax.set_title("Overall Rating vs Mentality Composure")
ax.set_ylabel('Mentality Composure Rating')
ax.set_xlabel('Overall Rating')

plt.show()

#top 15 Potential Vs Wages
fig, ax = plt.subplots(figsize=(8, 5))

pf.topNScatter(top_15, 'potential', 'wage_eur', 'short_name')

ax.set_title("Potential vs Wages of top 15")
ax.set_ylabel('Wages in Eur')
ax.set_xlabel('Potential')

plt.show()

print(df.groupby("work_rate")["wage_eur"].max())

print(df['work_rate'].unique())

#Wages & Level of Work Rate Bar Graph
fig, ax = plt.subplots(figsize=(7, 5))
df.groupby("work_rate")["wage_eur"].max().plot.bar()
plt.xlabel("Work Rate")
plt.ylabel("Wages in EUR")
plt.title("Wages by level of Workrate", fontsize=15)
plt.show()

#Hist Graph wages, Age, Height, Weight & no of players
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
#Below 2 using matlab
axes[0, 0].hist(df['wage_eur'])
axes[0, 0].set_xlabel('Wages in Euro')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Distribution of Wages in Euros')

axes[0, 1].hist(df['age'], bins = 15)
axes[0, 1].set_xlabel('Age of Players')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Distribution of Players Ages')

#below are using seaborn
axes[1, 0].set_title('Distribution of Height of Players')
sns.histplot(df, x='height_cm', ax=axes[1, 0], kde=True)
axes[1, 0].set_xlabel('Height in Centimeters')
axes[1, 0].set_ylabel('Count')


axes[1, 1].set_title('Distribution of Weight of Players')
sns.histplot(df, x='weight_kg', ax=axes[1, 1], kde=True)
axes[1, 1].set_xlabel('Weight in kg')
axes[1, 1].set_ylabel('Count')


plt.tight_layout(pad=2)
plt.show()

print(df.groupby(['preferred_foot']).count()[['sofifa_id']])

print(df.groupby(['preferred_foot']).count().sum()[['sofifa_id']])

#Pie Graph for preferred foot Right or Left
preferred_foot_labels = df["preferred_foot"].value_counts().index # (Right,Left)
preferred_foot_values = df["preferred_foot"].value_counts().values # (Right Values, Left Values)
explode = (0, 0.1) # used to separate the pie

# Visualize

plt.figure(figsize=(7, 7))
plt.pie(preferred_foot_values, labels=preferred_foot_labels, autopct='%1.2f%%')
plt.title('Football Players Preferred Feet', color='black', fontsize=15)
plt.legend()
plt.show()

print(df['nationality_name'].value_counts())

#Word Cloud of Country and count of players from tht country
nationality_name = " ".join(n for n in df['nationality_name'])
plt.figure(figsize=(10, 10))
wc = WordCloud().generate(nationality_name)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#bar Graph of top common countries vs count of players
bar_plot = dict(Counter(df['nationality_name'].values).most_common(5))
print(bar_plot)


fig, ax = plt.subplots(figsize=(8, 5))
plt.bar(*zip(*bar_plot.items()))
ax.set_title('Most Popular Nationalities')
plt.show()

def plot_most_common(category):
 bar_plot = dict(Counter(df[category].values).most_common(5))
 plt.bar(*zip(*bar_plot.items()))
 plt.show()

#bar Graph of top common countries vs count of players
fig, ax = plt.subplots(figsize=(8, 5))
plot_most_common('player_positions')
plt.show()

player_name = df[['wage_eur', 'short_name',
                  'value_eur', 'overall', 'age',
                  'nationality_name', 'potential',
                  'international_reputation']].nlargest(10, ['wage_eur']).set_index('short_name')
print(player_name)

# We get the names and overalls from the data
Overall = df["overall"]
footballer_name = df["short_name"]

# Let's create dataframe(Name,Overall)
data = pd.DataFrame({'short_name': footballer_name, 'overall': Overall})

x = df['short_name'].head(20)
y = df['overall'].head(20)

#barGraph top  20 players vs Ratings
plt.figure(figsize=(7, 10))

ax = sns.barplot(x=y, y=x, palette='Blues_r', orient='h')
plt.xticks()
plt.xlabel('Overall Ratings', size=20)
plt.ylabel('Player Names', size=20)
plt.title('FIFA22 Top 20 (Overall Rating)')

plt.show()
