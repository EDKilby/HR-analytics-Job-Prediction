#!/usr/bin/env python
# coding: utf-8

# # Welcome to the "HR Analytics Job Prediction" project! 
# 
# As a data analyst, you've been tasked by your stakeholders to investigate the responses to an HR survey distributed to each employee at your imaginary company. Utilize all tools available to you to investigate the determinants of exployee retention.
# 
# Let's start by importing our necessary libraries and reading in our dataset. 

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay,\
confusion_matrix, f1_score, classification_report


# In[2]:


# import dataset from Github "HR Dataset" branch
file_path = r"C:\Users\evank\OneDrive - University of North Florida\GitHub\HR_comma_sep.csv"
df = pd.read_csv(file_path)
df.head()


# # Exploratory Data Analysis

# Now let's explore our dataset with some basic EDA. I've provided a few examples oh how to use EDA to investigate our data. 
# Step 1) Gather information on datatypes of our columns
# Step 2) Harmonize our columns
# Step 2) Check for missing values. If any exist, consider how they should be handled.
# Step 3) Use descriptive statistics to examine our data. I've provided several methods of doing so.

# In[3]:


#Gather information on column datatypes
df.info()


# In[4]:


#Use a dictionary to rename columns that are not harmonized with format
df = df.rename(columns={
    'Work_accident': 'work_accident',
    'average_montly_hours': 'average_monthly_hours',
    'time_spend_company': 'tenure',
    'Department': 'department'})
df.columns


# In[5]:


#investigate how many values are missing
df.isna().sum()


# In[6]:


#investigate how many entries are duplicate values
print("Total Duplicate Entries:", df.duplicated().sum())
print('------------------------'*5)
df[df.duplicated].head()


# In[7]:


#drop all of the duplicate entries and create a new dataframe
df1 = df.drop_duplicates(keep="first")


# In[8]:


print(df1['department'].value_counts())
print("Department Count Total:", df1['department'].value_counts().sum(), end="\n\n")
print("------------"*5)
      
print(df1['salary'].value_counts())
print("Salary Count Total:", df1['salary'].value_counts().sum(), end='\n\n')      
print("------------"*5)
      
      
print(df1['work_accident'].value_counts())
print("Work_accident Count Total:", df1['work_accident'].value_counts().sum())  


# In[9]:


df1.describe()


# In[10]:


print(df1.groupby('left')['satisfaction_level'].mean())
print("------------"*5, end='\n\n')
print(df1.groupby('department')['left'].mean())


# # Check For Outliers

# In[11]:


#check for outliers in each column using boxplots 
cols = ['satisfaction_level','last_evaluation', 'number_project','average_monthly_hours', 'tenure']
for col in cols:
    plt.figure(figsize=(6,6))
    plt.title(f'Boxplot to detect outliers for {col}', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.boxplot(x=df1[col])
    plt.xlabel(col, fontsize=12)
    plt.show()


# Notice anything? There are some outliers in a certain column... tenure. 

# In[12]:


#Investigate how many outliers there are in the 'tenure' column
percentile25 = df1['tenure'].quantile(0.25)
percentile75 = df1['tenure'].quantile(0.75)
IQR = percentile75 - percentile25

upper_limit = percentile75 + 1.5 * IQR
lower_limit = percentile25 -1.5 * IQR
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]

print("Number of rows in data containing outliers in 'tenure':", len(outliers))


# As we move into the building phase of our machine learning model, consideration should be given to whether or not we drop these outliers: some models are more sensitive to outliers.

# # Data Visualizations

# In[13]:


fig, ax= plt.subplots(1,2, figsize= (22,8))

#boxplot: 'average_monthly_hours' distribution for 'number_project', comparing employees who stayed vs left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient='h', ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by # of projects worked on', fontsize='14')

#Histogram: distribution of 'number_project', comparing employees who stayed vs left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects Histogram', fontsize= '14')

plt.show()


# Breakdown of data visualization #1:
# 1) Everyone with 7 projects left the company, and the interquartiel ranges of this group and those who left with 6 projects was ~255-295 hours/month - much higher than their peers.
# 2) The optimal # of projects for employees work work on seems to be 3-4. The ratio of left/stayed is very small for these cohorts.
# 3) assuming normal work week of 40 hours and two weeks of vacation per year, the average number of working hours per month of employees working Monday-Friday = 50 weeks * 40 hours per week / 12 months = 166.67 hours per month. This means that, aside from the employees who worked on two projects - even those who didn't leave the company - worked considerably more hours than this. It seems these employees are overworked.

# In[14]:


#confirm that all employees who had 7 projects left
counts = df1[df1['number_project'] == 7]['left'].value_counts()
print("Stayed:", counts.get(0, 0))
print("Left:", counts.get(1, 0))


# We've now confirmed that all employees who has a case-load of 7 projects left.
# 
# Next, we'll examine the average monthly versus the satisfaction levels

# In[15]:


default_colors = sns.color_palette()

# Set up figure
plt.figure(figsize=(16, 9))

# Scatter plot
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)

# Vertical line
plt.axvline(x=166.67, color='k', ls='--')

# Custom legend handles
line_ = mlines.Line2D([], [], color='k', linestyle='--', label='166.67 hrs./mo.')
stay_ = mpatches.Patch(color=default_colors[0], label='Stayed')
left_ = mpatches.Patch(color=default_colors[1], label='Left')

# Apply custom legend
plt.legend(handles=[line_, stay_, left_])

# Title and labels
plt.title('Monthly hours by last evaluation score', fontsize=14)
plt.xlabel('average_monthly_hours')
plt.ylabel('satisfaction_level')
plt.show()


# Breakdown of data visualization #2
# 
# There is a considerable amount of employees who worked ~220-320 hours per month. 320 hours per month is 76.8 hours per week. If these hours are constitently logged across an entire year, it's not surprising that this cohort's satisfaction level is close to zero. 
# 
# Additionally, there is a cohort who left that had much more normal working hours (cluster of 'left' with average satisfaction level of 0.4). While it's difficult to speculate why this cohort left. They may have felt pressure to work more to keep up with their peers who worked longer hours, ultimately resulting in lower satisfaction levels within this cohort.
# 
# We will now visualize satisfaction levels by tenure

# In[16]:


fig, ax = plt.subplots(1,2, figsize=(22,8))

sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient='h', ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')

plt.show();


# Breakdown of data visualization #3
# 
# 1) There are two categories of employees who left: very satisfied employees with moderately long tenure and dissatisfied employees with shorter tenure
# 2) four-year employees appear to have an unusually low satisfaction level. 'promotion_last_5years' suggests that there be company policies that make promotions a timely endeavor. If possible, it's worth investigating changes to company policies that may affect people at the four-year tenure mark.
# 3) The longest employees didn't leave. Their satisfaction levels are on par with newer employees
# 4) the histogram reveals that there are very few highly-tenured employees. 
# 
# How about we quickly calculate the mean and median satisfaction scores of employees who left vs stayed similar to our early EDA

# In[17]:


df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])


# The mean and median satisfaction scores of employees who left were lower than those who stayed (as expected). We also see the mean satisfaction score appears to be slightly lower than the median in those who stay. This suggests that satisfaction_score may be skewed to the left, suggesting that there is a considerably sized group of dissatisfied employees that are pulling the mean down.
# 
# Now we'll examine salary levels for different tenures.

# In[18]:


fig, ax= plt.subplots(1,2, figsize = (22,8))

short_tenure = df1[df1['tenure'] < 7]
long_tenure = df1[df1['tenure'] > 6]

sns.histplot(data=short_tenure, x='tenure', hue='salary', discrete=1,
            hue_order=['low', 'medium', 'high'], multiple ='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people',fontsize='14')

sns.histplot(data=long_tenure, x='tenure', hue='salary', discrete=1,
            hue_order=['low', 'medium', 'high'], multiple ='dodge', shrink=.4, ax=ax[1])
ax[0].set_title('Salary histogram by tenure: long-tenured people', fontsize='14');        


# We can conclude that the longer-tenured employees were not disproproportionately comprised of higher-paid employees

# In[19]:


plt.figure(figsize=(16,4))
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='k', ls='--')

# Custom legend handles
line1 = mlines.Line2D([], [], color='k', linestyle='--', label='166.67 hrs./mo.')
stay1 = mpatches.Patch(color=default_colors[0], label='Stayed')
left1 = mpatches.Patch(color=default_colors[1], label='Left')
plt.legend(handles=[line1,stay1, left1])

plt.title("Monthly Hours by Promotion Within Last 5 Years")
plt.xlabel("average_monthly_hours")
plt.ylabel("promotion_last_5years")
plt.show()


# Breakdown of data visualization #5
# 
# 1) Very few employees who were promoted within the past 5 years left the company
# 2) Very few promotions across the past years
# 3) A majority of the employees who left were working the longest hours
# 4) Very few employees who worked the most hours were promoted
# 
# Let's now perform some department-based analyses

# In[20]:


df1['department'].value_counts()


# In[21]:


#Instead of creating a typical histogram like we've done before, let's create a %-based figure (instead of count-based)
crosstab = pd.crosstab(df1['department'], df1['left'], normalize='index') * 100
crosstab = crosstab[[0, 1]]
crosstab.plot(kind='bar', stacked=True, figsize=(12, 7), color=['#1f77b4', '#ff7f0e'])
plt.ylabel('Percentage')
plt.title('Distribution of Stayed vs Left by Department (Percent)', fontsize=14)
plt.legend(title='Left', labels=['Stayed', 'Left'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# There doesn't appear to be any department that significantly differs in its proportion of employees who left vs stayed. 
# 
# Finally, let's create a correlation matrix to analyze the correlation between our variables.

# In[22]:


df_corr = df1.select_dtypes(include='number')

plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(
    df_corr.corr(),
    vmin=-1,
    vmax=1,
    annot=True,
    cmap=sns.color_palette('vlag', as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 14}, pad=12)
plt.show()


# This confirms that a few relationships exist
# 1) Number of projects and average monthly hours has a relationship with evaluation scores
# 2) Number of projects has a positive relationship with average monthly hours
# 3) whether an employee leaves is negatively correlated wth their satisfaction level

# # General Insights From Analysis Phase

# It appears that employees are leaving this company due to working too many hours, being assigned too many projects, and evaluations not resulting in promotion. It's not a stretch to say that many of these employees likely feel burnout, but if they've been with the company for several years (>5 years), they are not likely to leave the company (not as a factor of being payed more). Additionally, there does not appear to be one department that is facing high and disproportionate levels of attrition.

# ## Machine Learning

# Now we can dive into the machine learning components of this project!

# In[23]:


df_enc = df1.copy()

# Clean and encode salary
df_enc['salary'] = df_enc['salary'].str.strip().str.lower()
df_enc['salary'] = pd.Categorical(
    df_enc['salary'],
    categories=['low', 'medium', 'high'],
    ordered=True
).codes

# Dummy encode 'department'
df_enc = pd.get_dummies(df_enc, columns=['department'], drop_first=False)

# Convert dummy bools to integers
df_enc = df_enc.astype({col: int for col in df_enc.columns if df_enc[col].dtype == bool})

df_enc.head()


# In[24]:


#create heatmap/correlation matrix to understand correlations between variables
plt.figure(figsize=(8,6))
sns.heatmap(df_enc[['satisfaction_level','last_evaluation','number_project','average_monthly_hours','tenure']]
           .corr(),annot=True, cmap='crest')
plt.title('Heatmap of dataset')
plt.show()


# In[25]:


#remove outliers from the tenure column that we identified earlier
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]
df_logreg.head()


# In[26]:


y= df_logreg['left']
print(y.head())
print('------------------------'*5)
X = df_logreg.drop('left', axis=1)
X.head()


# In[29]:


#splitting our data in training and testing data for the model
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=42)


# In[30]:


#Officially constructing the logistic regression model and fitting it with our data
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)


# In[31]:


#Use the logistic regression model to make predictions based on the testing set we defined two lines back.
y_pred = log_clf.predict(X_test)


# In[32]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, 
                                  display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot(values_format='')

# Display plot
plt.show()


# cheat sheet for the confusion matrix: 
# 
#     upper-left quadrant: true negatives
#     upper-right quadrant: false positives
#     lower-left quadrant: false negatives
#     lower-right quadrant: True positives
# 

# In[34]:


classifications = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=classifications))


# In[ ]:




