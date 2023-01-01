import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math

file = 'dataset/players_22.csv'
df = pd.read_csv(file)
print(df.head(5))

print(df.columns)

print(df.select_dtypes(include='number'))

print(df.shape)

print(df.dtypes)

print(df.describe())

plt.figure(figsize=(7,5))
ax = sns.scatterplot(x=df['potential'], y=df['wage_eur'])
plt.xlabel("Potential")
plt.ylabel("Wage(EUR)")
plt.title("Potential & Wage", fontsize=15)
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
plt.scatter(x=df['potential'], y=df['value_eur'])
plt.xlabel("potential")
plt.ylabel("Value in EUR")
plt.title("Potential & Value in EUR", fontsize=15)
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
plt.scatter(x=df['international_reputation'], y=df['value_eur'] )
plt.xlabel("International Reputation")
plt.ylabel("Value in EUR")
plt.title("Reputation & Value in EUR", fontsize = 15)
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
plt.scatter(x=df['international_reputation'], y=df['wage_eur'] )
plt.xlabel("International Reputation")
plt.ylabel("Wage in EUR")
plt.title("Reputation & wages in EUR", fontsize = 15)
plt.show()

fig, ax = plt.subplots(figsize = (8,5))
ax = sns.scatterplot(x =df['height_cm'], y = df['potential'])
plt.xlabel("Height")
plt.ylabel("Potential")
plt.title("Relationship between Height and Potential", fontsize = 16)
plt.show()

top_15 = df.nlargest(15, 'overall')
print(top_15.head(5))

fig, ax = plt.subplots(figsize=(8,5))

plt.scatter(top_15['overall'], top_15['mentality_composure'])

plt.text(top_15.iloc[0]['overall'],
top_15.iloc[0]['mentality_composure'],
top_15.iloc[0]['short_name'])

plt.text(top_15.iloc[1]['overall'],
 top_15.iloc[1]['mentality_composure'],
 top_15.iloc[1]['short_name'])

plt.text(top_15.iloc[2]['overall'],
 top_15.iloc[2]['mentality_composure'],
 top_15.iloc[2]['short_name'])
plt.text(top_15.iloc[3]['overall'],
 top_15.iloc[3]['mentality_composure'],
 top_15.iloc[3]['short_name'])
plt.text(top_15.iloc[4]['overall'],
 top_15.iloc[4]['mentality_composure'],
 top_15.iloc[4]['short_name'])
plt.text(top_15.iloc[5]['overall'],
 top_15.iloc[5]['mentality_composure'],
 top_15.iloc[5]['short_name'])
plt.text(top_15.iloc[6]['overall'],
 top_15.iloc[6]['mentality_composure'],
 top_15.iloc[6]['short_name'])
plt.text(top_15.iloc[7]['overall'],
 top_15.iloc[7]['mentality_composure'],
 top_15.iloc[7]['short_name'])
plt.text(top_15.iloc[8]['overall'],
 top_15.iloc[8]['mentality_composure'],
 top_15.iloc[8]['short_name'])
plt.text(top_15.iloc[9]['overall'],
 top_15.iloc[9]['mentality_composure'],
 top_15.iloc[9]['short_name'])

ax.set_title("Overall Rating vs Mentality Composure")
ax.set_ylabel('Mentality Composure Rating')
ax.set_xlabel('Overall Rating')

plt.show()

fig, ax = plt.subplots(figsize=(8,5))

plt.scatter(top_15['potential'], top_15['wage_eur'] )
plt.text(top_15.iloc[0]['potential'],
top_15.iloc[0]['wage_eur'], top_15.iloc[0]['short_name'])
# plt.text(top_15.iloc[1]['potential'], top_15.iloc[1]['wage_eur'], top_15.iloc[1]['short_name']) for better view
plt.text(top_15.iloc[2]['potential'],
 top_15.iloc[2]['wage_eur'], top_15.iloc[2]['short_name'])
# plt.text(top_15.iloc[3]['potential'], top_15.iloc[3]['wage_eur'], top_15.iloc[3]['short_name'])
plt.text(top_15.iloc[4]['potential'],
 top_15.iloc[4]['wage_eur'], top_15.iloc[4]['short_name'])

plt.text(top_15.iloc[5]['potential'],
 top_15.iloc[5]['wage_eur'], top_15.iloc[5]['short_name'])

plt.text(top_15.iloc[6]['potential'],
 top_15.iloc[6]['wage_eur'], top_15.iloc[6]['short_name'])

plt.text(top_15.iloc[7]['potential'],
 top_15.iloc[7]['wage_eur'], top_15.iloc[7]['short_name'])

plt.text(top_15.iloc[8]['potential'],
 top_15.iloc[8]['wage_eur'], top_15.iloc[8]['short_name'])

plt.text(top_15.iloc[9]['potential'],
 top_15.iloc[9]['wage_eur'], top_15.iloc[9]['short_name'])

ax.set_title("Potential vs Wages of top 15")
ax.set_ylabel('Wages in Eur')
ax.set_xlabel('Potential')

plt.show()

print(df.groupby("work_rate")["wage_eur"].max())

print(df['work_rate'].unique())

fig, ax = plt.subplots(figsize=(7,5))
df.groupby("work_rate")["wage_eur"].max().plot.bar()
plt.xlabel("Work Rate")
plt.ylabel("Wages in EUR")
plt.title("Wages by level of Workrate", fontsize = 15)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes[0,0].hist(df['wage_eur'])
axes[0,0].set_xlabel('Wages in Euro')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Distribution of Wages in Euros')

axes[0,1].hist(df['age'], bins = 15)
axes[0,1].set_xlabel('Age of Players')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Distribution of Players Ages')

# first two is using a matplotlib syntax, the next two I'll do with seaborn

axes[1,0].set_title('Distribution of Height of Players')
sns.histplot(df, x='height_cm', ax=axes[1,0], kde=True)
axes[1,0].set_xlabel('Height in Centimeters')
axes[1,0].set_ylabel('Count')


axes[1,1].set_title('Distribution of Weight of Players')
sns.histplot(df, x='weight_kg', ax=axes[1,1], kde=True)
axes[1,1].set_xlabel('Weight in kg')
axes[1,1].set_ylabel('Count')


plt.tight_layout(pad=2)
plt.show()

print(df.groupby(['preferred_foot']).count()[['sofifa_id']])

print(df.groupby(['preferred_foot']).count().sum()[['sofifa_id']])

preferred_foot_labels = df["preferred_foot"].value_counts().index # (Right,Left)
preferred_foot_values = df["preferred_foot"].value_counts().values # (Right Values, Left Values)
explode = (0, 0.1) # used to separate the pie

# Visualize

plt.figure(figsize = (7,7))
plt.pie(preferred_foot_values,
 labels=preferred_foot_labels,autopct='%1.2f%%')
plt.title('Football Players Preferred Feet',
color = 'black',fontsize = 15)
plt.legend()
plt.show()

print(df['nationality_name'].value_counts())

nationality_name = " ".join(n for n in df['nationality_name'])
plt.figure(figsize=(10, 10))
wc = WordCloud().generate(nationality_name)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

from collections import Counter
bar_plot = dict(Counter(df['nationality_name']
.values).most_common(5))
print(bar_plot)

fig, ax = plt.subplots(figsize = (8,5))
plt.bar(*zip(*bar_plot.items()))
ax.set_title('Most Popular Nationalities')
plt.show()


def plot_most_common(category):
 bar_plot = dict(Counter(df[category].values).most_common(5))
 plt.bar(*zip(*bar_plot.items()))
 plt.show()

fig, ax = plt.subplots(figsize = (8,5))
plot_most_common('player_positions')
plt.show()

player_name = df[['wage_eur','short_name',
                  'value_eur','overall','age',
                  'nationality_name','potential',
                  'international_reputation']].nlargest(10,['wage_eur']).set_index('short_name')
print(player_name)

# We get the names and overals from the data
Overall = df["overall"]
footballer_name = df["short_name"]

# Let's create dataframe(Name,Overall)
data = pd.DataFrame({'short_name': footballer_name,'overall':Overall})

x = df['short_name'].head(20)
y = df['overall'].head(20)

# plot
plt.figure(figsize=(7,10))


ax= sns.barplot(x=y, y=x, palette = 'Blues_r', orient='h')
plt.xticks()
plt.xlabel('Overall Ratings', size = 20)
plt.ylabel('Player Names', size = 20 )
plt.title('FIFA22 Top 20 (Overall Rating)')

plt.show()

