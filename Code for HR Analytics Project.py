#!/usr/bin/env python
# coding: utf-8

# # Welcome to the "HR Analytics Job Prediction" project! 
# 
# As a data analyst, you've been tasked by stakeholders to investigate the responses to an HR survey distributed to each employee at your company. Utilize all tools available to you to investigate the determinants of exployee retention and attrition.
# 
# Let's start by importing our necessary libraries and reading in our dataset. 

# In[1]:


#import libraries

#data manipulation
import numpy as np
import pandas as pd

#data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

#data modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#additional data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

#reporting metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay,\
confusion_matrix, f1_score, classification_report, roc_auc_score,roc_curve
from sklearn.tree import plot_tree


# In[2]:


# import dataset from Github "HR Dataset" branch
file_path = r"_______________________" #fill in the quotation marks with the actual file address
df = pd.read_csv(file_path)
df.head()


# # Exploratory Data Analysis

# Now let's explore our dataset with some basic exploratory data analysis (EDA). I've provided a few examples of how to use EDA to investigate our data.
# 
# Step 1) Gather information on datatypes of our columns
# 
# Step 2) Harmonize our columns
# 
# Step 2) Check for missing or duplicate values. If any exist, consider how they should be handled.
# 
# Step 3) Use descriptive statistics to examine our data. I've provided several methods of doing so.

# In[3]:


#Gather information on column datatypes. This is step 1) Gather information on datatypes of our columns
df.info()


# In[4]:


#Use a dictionary to rename columns that are not harmonized with the standard format. This is step 2) harmonize our columns
df = df.rename(columns={
    'Work_accident': 'work_accident',
    'average_montly_hours': 'average_monthly_hours',
    'time_spend_company': 'tenure',
    'Department': 'department'})
df.columns


# In[5]:


#investigate how many values are missing. This is part one of step 3) check for missing values
df.isna().sum()


# In[6]:


#investigate how many entries are duplicate values. This is part two of step 3) check for duplicate values
print("Total Duplicate Entries:", df.duplicated().sum())
print('------------------------'*5)
df[df.duplicated].head()


# Instead of performing a likelihood analysis using Bayes' theorem to determine the probability that these rows that contain duplicated values are in fact unreal entries, we can assume that the rows are more than likely to be fake because they
# 1) have the same exact values for 5 continuous variable columns
# 2) have the same exact values across 10 total columns 
# 
# Therefore, these rows should be dropped. These rows account for ~20% of our total data.

# In[7]:


#drop all of the duplicate entries and create a new dataframe
df1 = df.drop_duplicates(keep="first")


# In[8]:


#calculate proportions across nominal, ordinal, and boolean values. You can add more variables if you wish
print(df1['department'].value_counts())
print("Department Count Total:", df1['department'].value_counts().sum(), end="\n\n")
print("------------"*5)
      
print(df1['salary'].value_counts())
print("Salary Count Total:", df1['salary'].value_counts().sum(), end='\n\n')      
print("------------"*5)
      
print(df1['work_accident'].value_counts())
print("Work_accident Count Total:", df1['work_accident'].value_counts().sum())  

print(df1['left'].value_counts())


# In[9]:


#.describe() is the ultimate descriptive statistics function. Very helpful for quickly gathering information on columns
df1.describe()


# In[10]:


#determine average satisfaction levels for employees who stayed vs left
print(df1.groupby('left')['satisfaction_level'].mean())
print("------------"*5, end='\n\n')

#determine proportion of employees who left for each department category
print(df1.groupby('department')['left'].mean())


# # Check For Outliers

# In[11]:


#check for outliers in each column using boxplots 
cols = ['satisfaction_level','last_evaluation', 'number_project','average_monthly_hours', 'tenure']
for col in cols:
    plt.figure(figsize=(6,6))
    plt.title(f'Boxplot to Detect Outliers for {col}', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.boxplot(x=df1[col])
    plt.xlabel(col, fontsize=12)
    plt.show()


# Notice anything? There are some outliers in a certain column... tenure. 

# In[12]:


#Investigate how many outliers there are in the 'tenure' column. 
#Later we can delete these by deleting values outside upper and lower limits
percentile25 = df1['tenure'].quantile(0.25)
percentile75 = df1['tenure'].quantile(0.75)
IQR = percentile75 - percentile25

upper_limit = percentile75 + 1.5 * IQR
lower_limit = percentile25 -1.5 * IQR
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]

print("Number of rows in data containing outliers in 'tenure':", len(outliers))


# As we move into the building phase of our machine learning models, consideration should be given to whether or not we drop these outliers: some models are more sensitive to outliers.

# # Data Visualizations
# 
# Using various visualization techniques can be extremely helpful for understanding the relationships between our data. See comments denoted by "#" to see how we will be investigating relationships.  

# In[13]:


fig, ax= plt.subplots(1,2, figsize= (22,8))

#boxplot: 'average_monthly_hours' distribution for 'number_project', comparing employees who stayed vs left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient='h', ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly Hours by # of Projects Worked on', fontsize='14')

#Histogram: distribution of 'number_project', comparing employees who stayed vs left
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of Projects Histogram', fontsize= '14')

plt.show()


# Breakdown of data visualization #1:
# 1) Everyone with 7 projects left the company, and the interquartile ranges of this group and those who left with 6 projects was ~255-295 hours/month - much higher than their peers.
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

# Vertical line (representing 166.67 hrs./mo.)
plt.axvline(x=166.67, color='k', ls='--')

# Custom legend handles
line_ = mlines.Line2D([], [], color='k', linestyle='--', label='166.67 hrs./mo.')
stay_ = mpatches.Patch(color=default_colors[0], label='Stayed')
left_ = mpatches.Patch(color=default_colors[1], label='Left')

# Apply custom legend
plt.legend(handles=[line_, stay_, left_])

# Title and labels
plt.title('Monthly hours by Satisfaction Level', fontsize=14)
plt.xlabel('Average Monthly Hours')
plt.ylabel('Satisfaction Level')
plt.show()


# Breakdown of data visualization #2
# 
# There is a considerable amount of employees who worked ~220-320 hours per month. 320 hours per month is 76.8 hours per week. If these hours are consistently logged across an entire year, it's not surprising that this cohort's satisfaction level is close to zero. 
# 
# Additionally, there is a cohort of employees who left that had much more normal working hours (cluster of 'left' with average satisfaction level of 0.4). While it's difficult to speculate why this cohort left, they may have felt pressure to work more to keep up with their peers who worked longer hours, ultimately resulting in lower satisfaction levels within this cohort.
# 
# We will now visualize satisfaction levels by tenure

# In[16]:


fig, ax = plt.subplots(1,2, figsize=(22,8))

sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient='h', ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by Tenure Categories', fontsize='14')

sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure Histogram', fontsize='14')

plt.show();


# Breakdown of data visualization #3
# 
# 1) There are two categories of employees who left: very satisfied employees with moderately long tenure and dissatisfied employees with shorter tenure
# 2) four-year employees appear to have an unusually low satisfaction level. 'promotion_last_5years' suggests that there be company policies that make promotions a lengthy process and is just out of reach of the four-year employees. If possible, it's worth investigating changes to company policies that may affect people at the four-year tenure mark.
# 3) The longest employees didn't leave. Their satisfaction levels are on par with newer employees
# 4) the histogram reveals that there are very few highly-tenured employees. 
# 
# How about we quickly calculate the mean and median satisfaction scores of employees who left vs stayed similar to our early EDA

# In[17]:


#satisfaction levels (mean and median) across those who stayed vs left
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])


# The mean and median satisfaction scores of employees who left were lower than those who stayed (as expected). We also see the mean satisfaction score appears to be slightly lower than the median in those who stay. This suggests that satisfaction_score may be skewed to the left, suggesting that there is a considerably sized group of dissatisfied employees that are pulling the mean down.
# 
# Now we'll examine salary levels for different tenures.

# In[18]:


fig, ax= plt.subplots(1,2, figsize = (22,8))

#define what is long and short tenures
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


#Examine relationship between avg monthly hours, visual proportion of promoted vs not promoted, and relationship between promotion and left vs stayed

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


#We conducted a similar EDA earlier but you can use normalize=True to investigate the proportion of total departments counts
df1['department'].value_counts(normalize=True)


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

# # Machine Learning Phase

# ## Binomial Logistic Regression 

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


#create variables of interest for logistic regression model to predict (left). This is called feature selection
y= df_logreg['left']
print(y.head())
print('------------------------'*5)
X = df_logreg.drop('left', axis=1)
X.head()


# In[27]:


#splitting our data in training and testing data for the model
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=42)


# In[28]:


#Officially constructing the logistic regression model and fitting it with our data
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)


# In[29]:


#Use the logistic regression model to make predictions based on the testing set we defined two lines back.
y_pred = log_clf.predict(X_test)


# In[30]:


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

# In[31]:


#Get a report: performance metrics of our logistic regression model
classifications = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=classifications))


# Now we've successfully built a logistic regression model to predict when an employee is likely to leave the company based on the HR surveys. However, examining our performance metrics for a quick second would inform us that our model is moderately strong at predicting if an employee will stay, and very weak at predicting if an employee will stay. We can use other classification models to improve these metrics.
# 
# This next section will cover the construction of a decision tree model.

# # Decision Tree

# In[32]:


#feature selection
y = df_enc['left']
X = df_enc.drop('left', axis=1)

print(y.head())
print('--------------------'*6)
X.head(5)


# In[33]:


#split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# In[34]:


tree = DecisionTreeClassifier(random_state=42)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}

# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# In[35]:


get_ipython().run_cell_magic('time', '', 'tree1.fit(X_train, y_train)\n')


# In[36]:


#ok lets check the parameters that are best for our decision tree
tree1.best_params_


# In[37]:


#Determine the best AUC score on the cross validation
tree1.best_score_


# In[38]:


#Let's make a function that will help us extract all the scores from the grid search
def make_results(model_name:str, model_object, metric:str):
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'}
    cv_results = pd.DataFrame(model_object.cv_results_)
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]
    
    auc = best_estimator_results.mean_test_roc_auc
    f1= best_estimator_results.mean_test_f1
    recall= best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
    
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                         'precision': [precision],
                         'recall': [recall],
                         'F1': [f1],
                         'accuracy': [accuracy],
                         'auc': [auc]
                         })
    return table


# In[ ]:


#retrieve the cross-validation (CV) scores
tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
tree1_cv_results


# ## Random Forest Model

# In[40]:


rf = RandomForestClassifier(random_state= 42, class_weight='balanced')

#configure a dictionary of hyperparameters
cv_params = {'max_depth': [3,5,None],
             'max_features': [1,2, 'sqrt'],
             'max_samples': [0.7,1.0],
             'min_samples_leaf': [1,2],
             'n_estimators': [300, 500]}

#Configure a dictionary of scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

#incorporate GridSearch
rf1= GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit = 'roc_auc')


# In[41]:


get_ipython().run_cell_magic('time', '', 'rf1.fit(X_train, y_train)\n')


# In[42]:


#This will identify the best AUC score achieved the by decision tree
rf1.best_score_


# In[ ]:


#This code will pull the best parameters to use for this model. Helpful if you want to further tune the model.
rf1.best_params_


# In[44]:


#retrieve all CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)
print(rf1_cv_results)


# In[74]:


#defining a class that will pull results for all of our CV scores (testing data)
def get_scores(model_name: str, model, X_test_data, y_test_data):
    
    preds = model.best_estimator_.predict(X_test_data)
    
    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)
    
    table = pd.DataFrame({'model': [model_name],
                         'precision': [precision],
                         'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
    return table


# In[46]:


#we can now write to get our performance metrics report
rf1_test_scores = get_scores('random forest test', rf1, X_test, y_test)
rf1_test_scores


# # Feature Engineering
# 
# These are very good scores, but to be thorough, it would be best to include a feature engineering phase to ensure that there is no data leakage. Data leakage occurs when data implemented during the training phase is inappropriately used to train your model. This occurs for two reasons: 1) the data appears in the test data or 2) the data being used to train the model is data that would not be expected whenever we deploy the model. To ensure best model performance, we need to address this potential leaked data.  

# It's likely that the company will not have satisfaction levels reported for all employees upon time of employee departure. Additionally, "average_monthly_hours" may be contributing to data leakage. If employees have already decided to depart from the company, or management has identified individuals to be fired, those employees may be working less hours.

# We can drop satifaction_level and create a new column that roughly captures if an employee is overworked (in place of average_monthly_hours)

# The first rounds of the decision tree and the random forest models included all variables as features. Through feature engineering, we'll be able to build improved models

# In[47]:


df_feat_eng = df_enc.drop('satisfaction_level', axis=1)

df_feat_eng.head()


# In[48]:


df_feat_eng['overworked'] = df_feat_eng['average_monthly_hours']

print('Max hours:', df_feat_eng['overworked'].max())
print('Min hours', df_feat_eng['overworked'].min())


# We'll now transform the integer column "overworked" into a binary-based column using a boolean mask. Earlier in this project, we established that the normal amount of hours in a work month is 166.67 hours. To convert the continuous "overworked" column to binary, we can assign all values in "overworked" column to 1 when they are above 170, and we will assign all values to 0 when they are equal to or less than 170. You can play around with this threshold; we will be adjusting this threshold later to see how it affects performance

# In[49]:


df_feat_eng['overworked'] = (df_feat_eng['overworked'] > 175).astype(int)
df_feat_eng.head()


# now that we've transformed the "overworked" column, we can drop the average_monthly_hours column

# In[50]:


df_feat_eng = df_feat_eng.drop('average_monthly_hours', axis=1)


# ## Decision Tree Round 2

# In[51]:


y = df_feat_eng['left']
X = df_feat_eng.drop('left', axis=1)


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=42)


# In[53]:


tree = DecisionTreeClassifier(random_state = 42)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}

# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# In[54]:


get_ipython().run_cell_magic('time', '', 'tree2.fit(X_train, y_train)\n')


# In[55]:


tree2.best_params_


# In[56]:


tree2.best_score_


# In[57]:


tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
print(tree1_cv_results)
print(tree2_cv_results)


# While some of the scores fell, this is to be expected given fewer features were included in the second round of the decision tree model. All things considered, these are very good scores.

# # Random Forest - Round 2

# Sometimes it's extremely beneficial to rely on randomized CV searches to discover the optimal hyperparameters for your models. Otherwise, errors and wait times can be a big issue! These randomized searches can run quickly and can determine what hyperparameters are optimal

# In[58]:


#Before we begin defining hyperparameters, let's check the split of our training set
print(y_train.value_counts(normalize=True))


# In[59]:


# Second round: refined and slightly expanded hyperparameter space
cv_params = {
    'max_depth': [5,10,12],
    'max_features': ['sqrt', 'log2'],
    'max_samples': [0.7, 0.8, 1.0],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [300, 400, 500]
}


scoring = {
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'}

rf2 = RandomizedSearchCV(
    estimator=rf,
    param_distributions=cv_params,
    n_iter=12,
    scoring=scoring,
    refit='roc_auc',
    cv=4,
    verbose=1,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)


# In[60]:


get_ipython().run_cell_magic('time', '', 'rf2.fit(X_train, y_train) \n')


# In[61]:


rf2.best_params_


# In[62]:


rf_r2 = RandomForestClassifier(random_state=42, class_weight= 'balanced')

#configure a dictionary of hyperparameters
cv_params = {
    'max_depth': [10, 15],
    'max_features': ['sqrt', 'log2'],
    'max_samples': [0.8, 1.0],
    'min_samples_leaf': [2, 4],
    'min_samples_split': [4, 5],
    'n_estimators': [300, 400,]}

#configure a dictionary of scoring parameters
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'}

#incorporate GridSearch
rf3 = GridSearchCV(rf_r2, cv_params, scoring=scoring, cv=4, refit = 'roc_auc')


# In[63]:


get_ipython().run_cell_magic('time', '', 'rf3.fit(X_train, y_train)\n')


# In[64]:


rf3.best_params_


# In[65]:


rf3.best_score_


# In[66]:


rf3_cv_results = make_results('random forest2 cv', rf3, 'auc')
print(tree2_cv_results)
print(rf3_cv_results)


# In[67]:


rf3_test_scores = get_scores('random forest2 test', rf3,X_test, y_test)
rf3_test_scores


# plot a confusion matrix to visualize how well it predicts the test set

# In[68]:


predict = rf3.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, predict, labels = rf2.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf3.classes_)
disp.plot(values_format='')


# Now we'll plot the splits of our tree so we can understand how our decision tree operates

# In[69]:


#convert the pandas index to a list
feature_names= list(X.columns)


# In[75]:


#Visualize the splits of our second round decision tree
plt.figure(figsize=(85, 20))
plot_tree(
    tree2.best_estimator_,
    max_depth=6,
    fontsize=14,
    feature_names=feature_names,
    class_names=['stayed', 'left'],
    filled=True
)
plt.show()


# to get a closer view of this picture, double tap it and use the scrolling bar towards the bottom of the picture to see each split

# Now we'll implement feature importance to get insights into which variables contribute the most towards successfully predicting if an employee will leave or not.

# In[71]:


tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_,
                               columns= ['gini_importance'],
                               index=feature_names)
tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending= False)

#Only include features with importances greater than 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] !=0]
tree2_importances


# For the decision tree, 'number_project', 'last_evaluation', and 'tenure' contribute signficantly to the ability of the model to predict if an employee will leave the company

# In[76]:


#Visualize the feature importances for the decision tree
plt.figure(figsize=(5,5))
sns.barplot(data=tree2_importances, x='gini_importance', y=tree2_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Attrition", fontsize='14')
plt.ylabel("Feature", fontsize='12')
plt.xlabel('Importance', fontsize='13')
plt.show()


# We'll now compute feature importance for the random forest and compare how these differ from the importances from the decision tree

# In[80]:


# Get feature importances
importances = rf2.best_estimator_.feature_importances_

# Get indices of top 10 features (unsorted)
ind = np.argpartition(importances, -10)[-10:]

# Optionally sort those indices by importance
ind = ind[np.argsort(importances[ind])]

# Get the actual feature names and importance values
feat_names = X.columns[ind]
feat_importances = importances[ind]

# Create DataFrame
y_df = pd.DataFrame({"Feature": feat_names, "Importance": feat_importances})

# Plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
y_df.plot(kind='barh', ax=ax1, x='Feature', y='Importance')

ax1.set_title('Random Forest: Feature Importances for Attrition', fontsize=12)
ax1.set_ylabel('Feature')
ax1.set_xlabel('Importance')

plt.show()


# In this case, we see that the same 4 most important features are the same across the decision tree and random forest models, but tenure is the greatest determinant of attrition.

# # Project Overview

# The logistic regression model achieve a weighted average precision of 79%, recall of 82%, and accuracy of 82% on the testing data set. 
# 

# After feature engineering, the decision tree model achieved good performance metric scores:
# 
#     1) AUC of 95.9%
#     2) Precision of 79.2% 
#     3) Accuracy of 94.4%
#     4) F1 score of 84.3%
#     5) Recall of 90.2%

# Following feature engineering, the random forest modestly outperformed the decision tree model. The performance metric scores:
#     
#     1) AUC: 94.2%
#     2) Accuracy: 95.4%
#     3) Recall: 92.5%
#     4) Precision: 82.0%
#     5) F1 Score: 87.0%

# The visualizations, models, and features importances confirm that these employees at the company are overworked.
#     Potential foci for stakeholders include
#     
# 1) Cap the number of projects that employees can work on
# 
# 2) Consider investigating company policies to understand why four-year tenure employees are so dissatisfied, or consider re-evaluating company policies to allow for more frequent (with justification of performance metrics) promotions
# 
# 3) Consider re-evaluating company policies about overtime pay, reward employees who work longer hours, or do not make them work the long hours that most are.
# 
# 4) Inform employees about company policies surrounding overtime pay. If work expectations are not clear or have not been defined, hold a meeting to discuss and allow for questions
# 
# 5) During said meeting, take the time to discuss company culture, individual and team functionality, and listen to and address topics of concern for employees
# 
# 6) Re-evaluate performance metrics (not just evaluation scores and average monthly hours). Projects completed, ROI, and other metrics should carry more weight in the decision process for promotion

# ## Congrats on finishing the HR analytics project. You've made significant strides in tackling the attrition rate in the company, have built several impressive models, and have actionable items to bring to stakeholders

# In[ ]:




