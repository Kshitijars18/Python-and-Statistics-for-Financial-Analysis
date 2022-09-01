import pandas as pd   # pandas is fast flexible and expressive data structure
import numpy as np #Used for numerical computing of array and matrix
import matplotlib #ploting package
import statsmodels #for regression and time series analysis
# importing the required module
import matplotlib.pyplot as plt

col_names = ['Age','Gender','Plans_heard','Interested_Sector','Asset_Persisting','Demat','Perception','Opinion','Stock_Exchange','Sector','Returns_Expected','Time_Spend','Stockbroker','Tax_Benefit', 'Max_Profit','Strategies','Good_Investor','Risk','Manage_Risk','Blue_Chip_Stock','Device','Advantage','Disadvantage','Class']
st = pd.read_csv('/content/Revised data.csv', names=col_names)
#st.drop(["Timestamp","1) Your full name","2) Your email id or mobile number"], axis = 1, inplace = True)
Goal = st[['Age','Gender','Plans_heard','Interested_Sector','Asset_Persisting','Demat','Perception','Opinion','Stock_Exchange','Sector','Returns_Expected','Time_Spend','Stockbroker','Tax_Benefit', 'Max_Profit','Strategies','Good_Investor','Risk','Manage_Risk','Blue_Chip_Stock','Device','Advantage','Disadvantage','Class']]
print(Goal)
st['Age1']= pd.Categorical(st['Age'])
st['Age2'] = st.Age1.cat.codes
st['Gender1']=pd.Categorical(st['Gender'])
st['Gender2'] = st.Gender1.cat.codes
st['Plans_heard1']= pd.Categorical(st['Plans_heard'])
st['Plans_heard2'] = st.Plans_heard1.cat.codes
st['Interested_Sector1']=pd.Categorical(st['Interested_Sector'])
st['Interested_Sector2'] = st.Interested_Sector1.cat.codes
st['Asset_Persisting1']= pd.Categorical(st['Asset_Persisting'])
st['Asset_Persisting2'] = st.Asset_Persisting1.cat.codes
st['Demat1']=pd.Categorical(st['Demat'])
st['Demat2'] = st.Demat1.cat.codes
st['Perception1']= pd.Categorical(st['Perception'])
st['Perception2'] = st.Perception1.cat.codes
st['Opinion1']=pd.Categorical(st['Opinion'])
st['Opinion2'] = st.Opinion1.cat.codes
st['Stock_Exchange1']= pd.Categorical(st['Stock_Exchange'])
st['Stock_Exchange2'] = st.Stock_Exchange1.cat.codes
st['Sector1']=pd.Categorical(st['Sector'])
st['Sector2'] = st.Sector1.cat.codes
st['Returns_Expected1']= pd.Categorical(st['Returns_Expected'])
st['Returns_Expected2'] = st.Returns_Expected1.cat.codes
st['Time_Spend1']=pd.Categorical(st['Time_Spend'])
st['Time_Spend2'] = st.Time_Spend1.cat.codes
st['Stockbroker1']= pd.Categorical(st['Stockbroker'])
st['Stockbroker2'] = st.Stockbroker1.cat.codes
st['Tax_Benefit1']=pd.Categorical(st['Tax_Benefit'])
st['Tax_Benefit2'] = st.Tax_Benefit1.cat.codes
st['Max_Profit1']= pd.Categorical(st['Max_Profit'])
st['Max_Profit2'] = st.Max_Profit1.cat.codes
st['Strategies1']=pd.Categorical(st['Strategies'])
st['Strategies2'] = st.Strategies1.cat.codes
st['Good_Investor1']= pd.Categorical(st['Good_Investor'])
st['Good_Investor2'] = st.Good_Investor1.cat.codes
st['Risk1']=pd.Categorical(st['Risk'])
st['Risk2'] = st.Risk1.cat.codes
st['Manage_Risk1']= pd.Categorical(st['Manage_Risk'])
st['Manage_Risk2'] = st.Manage_Risk1.cat.codes
st['Blue_Chip_Stock1']=pd.Categorical(st['Blue_Chip_Stock'])
st['Blue_Chip_Stock2'] = st.Blue_Chip_Stock1.cat.codes
st['Device1']= pd.Categorical(st['Device'])
st['Device2'] = st.Device1.cat.codes
st['Advantage1']=pd.Categorical(st['Advantage'])
st['Advantage2'] = st.Advantage1.cat.codes
st['Disadvantage1']= pd.Categorical(st['Disadvantage'])
st['Disadvantage2'] = st.Disadvantage1.cat.codes
st['Class1']=pd.Categorical(st['Class'])
st['Class2'] = st.Class1.cat.codes

st.head()

st_dataset = st[['Age2','Gender2','Plans_heard2','Interested_Sector2','Asset_Persisting2','Demat2','Perception2','Opinion2','Stock_Exchange2','Sector2','Returns_Expected2','Time_Spend2','Stockbroker2','Tax_Benefit2', 'Max_Profit2','Strategies2','Good_Investor2','Risk2','Manage_Risk2','Blue_Chip_Stock2','Device2','Advantage2','Disadvantage2','Class2']]
st_feature = st[['Age2','Gender2','Plans_heard2','Interested_Sector2','Asset_Persisting2','Demat2','Perception2','Opinion2','Stock_Exchange2','Sector2','Returns_Expected2','Time_Spend2','Stockbroker2','Tax_Benefit2', 'Max_Profit2','Strategies2','Good_Investor2','Risk2','Manage_Risk2','Blue_Chip_Stock2','Device2','Advantage2','Disadvantage2']]
st_target = st[['Class2']]
print(st_feature)
print(st_target)

st.corr()

# To prevent the overfitting part
from sklearn.model_selection import train_test_split
st_feature_train, st_feature_test, st_target_train, st_target_test = train_test_split(
    st_feature,
    st_target,
    test_size = 0.2,
    random_state = 0)
st_feature_train.shape, st_feature_test.shape

############ Heatmap ############
import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize =(15,10))
cor = st_feature_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.YlGnBu)
plt.show()

def correlation(st_dataset,threshold):
  col_corr = set()
  corr_matrix = st_dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i,j])>threshold:
        colname = corr_matrix.columns[i]
        col_corr.add(colname)
  return col_corr
  
 corr_features = correlation(st_feature_train,0.7)
len(set(corr_features))

corr_features

st_feature_train.drop(corr_features, axis=1)
st_feature = st[['Age2','Gender2','Plans_heard2','Interested_Sector2','Asset_Persisting2','Demat2','Perception2','Opinion2','Stock_Exchange2','Sector2','Returns_Expected2','Time_Spend2','Stockbroker2','Tax_Benefit2', 'Max_Profit2','Strategies2','Good_Investor2','Risk2','Manage_Risk2','Blue_Chip_Stock2','Device2','Advantage2','Disadvantage2']]

########### Density plot of several variables ###############
# libraries & dataset
import seaborn as sns
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")

 
# plotting both distibutions on the same figure
fig = sns.kdeplot(st_feature['Time_Spend2'], shade=True, color="green")
fig = sns.kdeplot(st_feature['Returns_Expected2'], shade=True, color="yellow")
plt.show()

############ Histogram with several variables with Seaborn ############
import seaborn as sns
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")

sns.histplot(data=st_feature, x="Age2", color="skyblue", label="Age (Ranges from 18 to 55+ yrs)", kde=True)
sns.histplot(data=st_feature, x="Stock_Exchange2", color="red", label="Stock_Exchange(NSE, BSE, other)", kde=True)
plt.legend() 
plt.show()

############ Boxplot ###########
sns.boxplot( x=st_feature["Advantage2"])

############ Treemap ###########
!pip install squarify
import squarify
# libraries
import matplotlib.pyplot as plt
import squarify    # pip install squarify (algorithm for treemap)
import pandas as pd

# Create a data frame with fake data
df = pd.DataFrame({'investment':[49,46,31,33,45,35,1], 'group':["Mutual Funds", "Stock Market", "PPF", "Real Estate","Gold","FD","Other"] })

# plot it

squarify.plot(sizes=df['investment'], label=df['group'],color=["#E0BBE4","#DCFFFB","#FFDCF4", "#C1E7E5","#fdffb6","#caffbf","#a0c4ff"], alpha=.8 )
plt.axis('off')
plt.show()

# Create a data frame with fake data
df = pd.DataFrame({'sector':[42,36,32,21,22,35,18], 'group':["Bank", "Health Care", "Education", "Agriculture","Telecommunication","IT","Parmaceutical"] })

# plot it

squarify.plot(sizes=df['sector'], label=df['group'],color=["#FF9AA2","#E2F0CB","#DABFDE", "#fff1f1","#FFFFB5","#ABDEE6","#f8df81"], alpha=.8 )
plt.axis('off')
plt.show()

############ Pairplot ###########
# library & dataset
import matplotlib.pyplot as plt
import seaborn as sns

# without regression
sns.pairplot(st_feature, kind="scatter")
plt.show()

# with regression
sns.pairplot(st_feature, kind="reg")
plt.show()

############ Circular packaging ###########
!pip install circlify
df = pd.DataFrame({
    'Device': ['Tablet (14.8%)', 'Laptop (54.1%)', 'Smartphone (90.2%)'],
    'Value': [9,33,55]
})
# import the circlify library
import circlify

# compute circle positions:
circles = circlify.circlify(
    df['Value'].tolist(), 
    show_enclosure=False, 
    target_enclosure=circlify.Circle(x=0, y=0, r=1)
)
import circlify
import matplotlib.pyplot as plt

# Create just a figure and only one subplot
fig, ax = plt.subplots(figsize=(10,10))

# Title
ax.set_title('Device preference by responder')

# Remove axes
ax.axis('off')

# Find axis boundaries
lim = max(
    max(
        abs(circle.x) + circle.r,
        abs(circle.y) + circle.r,
    )
    for circle in circles
)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

# list of labels
labels = df['Device']

# print circles
for circle, label in zip(circles, labels):
    x, y, r = circle
    ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2,facecolor="#00FF00"))
    plt.annotate(
          label, 
          (x,y ) ,
          va='center',
          ha='center'
     )
    
df = pd.DataFrame({
    'Invest': [ 'Always invest your surplus funds (27.9%)','Invest for Dividends (31.1%)', 'Always have realistic goals (67.2%)','Have a disciplined approach for investment(72.1%)'],
    'Value': [19,17,41,44]
})
# import the circlify library
# compute circle positions:
circles = circlify.circlify(
    df['Value'].tolist(), 
    show_enclosure=False, 
    target_enclosure=circlify.Circle(x=0, y=0, r=1)
)


# Create just a figure and only one subplot
fig, ax = plt.subplots(figsize=(10,10))

# Title
ax.set_title('Strategies to be followed')

# Remove axes
ax.axis('off')

# Find axis boundaries
lim = max(
    max(
        abs(circle.x) + circle.r,
        abs(circle.y) + circle.r,
    )
    for circle in circles
)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

# list of labels
labels = df['Invest']

# print circles
for circle, label in zip(circles, labels):
    x, y, r = circle
    ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2, facecolor="#CD5C5C"))
    plt.annotate(
          label, 
          (x,y ) ,
          va='center',
          ha='center'
     )
     
 ############ WordCloud ###########
 from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a list of word
text=("'Research the company before investing' 'Invest According to risk' 'Diversification (spread your investments across multiple stocks)' 'Add some high-dividend paying stock to your portfolio' ")

# Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0, background_color="skyblue").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a list of word
text=("Demat Stock_Exchange Sector Returns_Expected Time_Spend Stockbroker Plans_heard Interested_Sector Asset_Persisting  Age Gender  Perception Opinion  Tax_Benefit Max_Profit Strategies2 Good_Investor2 Risk2 Manage_Risk Blue_Chip_Stock  Device Advantage2 Disadvantage Class2")

# Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0, background_color="yellow").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

############ Linear Model for Prediction ###########
import sklearn.linear_model
LR_model = sklearn.linear_model.LinearRegression()
x = st_target[['Class2']]
y = st_feature[['Perception2']]

LR_model.fit(x,y)
print("The Slope of the line is Beta1 ::", LR_model.coef_)
print("The Beta0 ::", LR_model.intercept_)
print(st_target[['Class2']])
y_pred = LR_model.predict([[1]])
print("Predicted value for the new input::", y_pred)
y_pred = LR_model.predict(x)
print("y actual",y)
print("y predict",y_pred)

   
