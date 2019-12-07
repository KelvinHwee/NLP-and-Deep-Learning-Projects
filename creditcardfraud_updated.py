#==============================================================================
#=====     setting some configurations
#==============================================================================
import os
path = r'C:\Users\eight\Desktop\Kelvin HDD\5. Kaggle\Kaggle competition\12. Credit Card Fraud Detection'
os.chdir(path)
os.getcwd()

import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

#==============================================================================
#=====     read in the anonymised credit card data
#==============================================================================
import pandas as pd
credit = pd.read_csv('creditcard.csv')
credit.head()

# since the data is related to fraud, we examine the proportion of the classes
import seaborn as sns
sns.countplot(credit.Class)
credit.Class.value_counts()

# define the feature columns and the target column
list(credit)
X = credit.iloc[:,:-1]; y = credit.iloc[:,-1]


#==============================================================================
#=====     apply SMOTE technique to over-sample the minority class
#==============================================================================
# note that we perform the oversampling before doing cross-validation
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# take a look at the shape of the datasets before application of SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("Shape of the X_train dataset: ", X_train.shape)
print("Shape of the y_train dataset: ", y_train.shape)
print("Shape of the X_test dataset: ", X_test.shape)
print("Shape of the y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

# we now implement the SMOTE technique
# note that we need to ensure that "y_train" is continuous; so we use the "ravel" method to do that
smote = SMOTE(random_state = 0)
X_train_oversample, y_train_oversample = smote.fit_sample(X_train, y_train.ravel())
print("Shape of the X_train dataset after over-sampling: ", X_train_oversample.shape)
print("Shape of the y_train dataset after over-sampling: ", y_train_oversample.shape)

# we now have a balanced dataset
print("Before OverSampling, counts of label '1': {}".format(sum(y_train_oversample == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_oversample == 0)))

import seaborn as sns
sns.countplot(y_train_oversample)


#==============================================================================
#=====     create some visualisations
#==============================================================================
# create the correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def corr_heatmap(arr):
	df = pd.DataFrame(arr)
	correlations = df.corr()
	cmap = sns.diverging_palette(220, 10, as_cmap = True)
	fig, ax = plt.subplots(figsize = (12,12))
	sns.heatmap(correlations, cmap = cmap, vmax = 1.0, 
			    center = 0, fmt = '.1f', square = True,
				linewidth = 0.5, annot = True, cbar_kws = {"shrink": .80})

	plt.show()

# we can see that after performing the over-sampling, that we see more correlations
corr_heatmap(X_train)
corr_heatmap(X_train_oversample)


#==============================================================================
#=====     clustering by PCA
#==============================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# select random four groups of 1000 samples each
y_df = pd.DataFrame(y_train_oversample)
y_df.rename({0: 'target'}, axis = 1, inplace = True)
X_df = pd.DataFrame(X_train_oversample)

# sample group 1
y_sample1 = y_df.sample(n = 1000, random_state = 0)
y_sample1.index
X_sample1 = X_df.iloc[y_sample1.index]

# sample group 2
y_sample2 = y_df.sample(n = 1000, random_state = 1)
y_sample2.index
X_sample2 = X_df.iloc[y_sample2.index]

# sample group 3
y_sample3 = y_df.sample(n = 1000, random_state = 2)
y_sample3.index
X_sample3 = X_df.iloc[y_sample3.index]

# sample group 4
y_sample4 = y_df.sample(n = 1000, random_state = 3)
y_sample4.index
X_sample4 = X_df.iloc[y_sample4.index]

# before applying PCA, each feature should be centered (zero mean) and with unit variance
# sample group 1
X_norm1 = StandardScaler().fit_transform(X_sample1)
X_pca1 = PCA(n_components = 2, random_state = 0).fit_transform(pd.DataFrame(X_norm1).values)

# sample group 2
X_norm2 = StandardScaler().fit_transform(X_sample2)
X_pca2 = PCA(n_components = 2, random_state = 0).fit_transform(pd.DataFrame(X_norm2).values)

# sample group 3
X_norm3 = StandardScaler().fit_transform(X_sample3)
X_pca3 = PCA(n_components = 2, random_state = 0).fit_transform(pd.DataFrame(X_norm3).values)

# sample group 4
X_norm4 = StandardScaler().fit_transform(X_sample4)
X_pca4 = PCA(n_components = 2, random_state = 0).fit_transform(pd.DataFrame(X_norm4).values)

# doing a PCA scatter plot
import seaborn as sns
fig, ax = plt.subplots(2, 2, figsize = (12,10))
sns.scatterplot(x = X_pca1[:,0], y = X_pca1[:,1], hue = (y_sample1.target == 1), 
					palette = ['blue', 'red'], alpha = 0.75, ax = ax[0,0])

sns.scatterplot(x = X_pca2[:,0], y = X_pca2[:,1], hue = (y_sample2.target == 1), 
					palette = ['blue', 'red'], alpha = 0.75, ax = ax[0,1])

sns.scatterplot(x = X_pca3[:,0], y = X_pca3[:,1], hue = (y_sample3.target == 1), 
					palette = ['blue', 'red'], alpha = 0.75, ax = ax[1,0])

sns.scatterplot(x = X_pca4[:,0], y = X_pca4[:,1], hue = (y_sample4.target == 1), 
					palette = ['blue', 'red'], alpha = 0.75, ax = ax[1,1])


#==============================================================================
#=====     we scale the variables to prevent overfitting
#==============================================================================
# initialise the MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler().fit(X_train_oversample[:,(0,-1)])

# create dataframes for both train and test
import pandas as pd
X_df_train = pd.DataFrame(X_train_oversample)
X_df_test  = pd.DataFrame(X_test)

# proceed to scale the data
X_df_train.iloc[:,[0,-1]] = mm_scaler.transform(X_df_train.iloc[:,[0,-1]])
X_df_test.iloc[:,[0,-1]]  = mm_scaler.transform(X_df_test.iloc[:,[0,-1]])

# convert back to numpy arrays
import numpy as np
X_mmscaled_train = np.asarray(X_df_train)
X_mmscaled_test  = np.asarray(X_df_test)


#==============================================================================
#=====     we perform some predictions using four classifiers
#==============================================================================
#===   load in the packages
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

#===   create the logistic regression classifier and compute accuracy scores
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
clf_logreg = LogisticRegression().fit(X_mmscaled_train, y_train_oversample)
logreg_pred = clf_logreg.predict(X_mmscaled_test)
confusion_matrix(y_test, logreg_pred)
print("The AUC score from Logistic Regression is: {}".format(roc_auc_score(y_test, logreg_pred)))
print("The classification report from Logistic Regression is: {}".format(classification_report(y_test, logreg_pred)))

#===   create the Support Vector Classifier and compute accuracy scores
clf_SVC = SVC().fit(X_mmscaled_train, y_train_oversample)
SVC_pred = clf_SVC.predict(X_mmscaled_test)
confusion_matrix(y_test, SVC_pred)
print("The AUC score from SVC is: {}".format(roc_auc_score(y_test, SVC_pred)))
print("The classification report from SVC is: {}".format(classification_report(y_test, SVC_pred)))

#===   create the RandomForest Classifier and compute accuracy scores
clf_rf = RandomForestClassifier().fit(X_mmscaled_train, y_train_oversample)
rf_pred = clf_rf.predict(X_mmscaled_test)
confusion_matrix(y_test, rf_pred)
print("The AUC score from Random Forest is: {}".format(roc_auc_score(y_test, rf_pred)))
print("The classification report from Random Forest is: {}".format(classification_report(y_test, rf_pred)))

#===   create the RandomForest Classifier and compute accuracy scores
clf_ada = AdaBoostClassifier().fit(X_mmscaled_train, y_train_oversample)
ada_pred = clf_ada.predict(X_mmscaled_test)
confusion_matrix(y_test, ada_pred)
print("The AUC score from AdaBoost is: {}".format(roc_auc_score(y_test, ada_pred)))
print("The classification report from AdaBoost is: {}".format(classification_report(y_test, ada_pred)))







from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators = 25, random_state = 0)
clf_rf.fit(X_train_oversample, y_train_oversample)
clf_rf.score(X_test, y_test)
y_pred = clf_rf.predict(X_test)

from sklearn.metrics import confusion_matrix, roc_auc_score
confusion_matrix(y_test, y_pred)
roc_auc_score(y_test, y_pred)

#===   consider the LogisticRegression classifier
from sklearn.linear_model import LogisticRegression
clf_logreg = LogisticRegression().fit(X_mmscaled, y_train_oversample)
clf_logreg.score(X_test, y_test)
y_pred_logreg = clf_logreg.predict(X_test)
confusion_matrix(y_test, y_pred_logreg)
roc_auc_score(y_test, y_pred_logreg)


# implementing SMOTE technique and cross validing the right way
for train, test in sss.split

sss.split(X,y)


































































