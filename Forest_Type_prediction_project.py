"""Problem Statement:
Forest land is highly required for developing ecosystem management. Any changes that
occur in ecosystem should be carefully noticed to avoid further loss. This model is
helpful in noticing the changes occurred due to heavy floods or any other calamities
which affected the forest land.
The goal is to predict seven different cover types in four different wilderness areas of
the Roosevelt National Forest of Northern Colorado with the best accuracy

Four wilderness areas are:
 1: Rawah
 2: Neota
 3: Comanche Peak
 4: Cache la Poudre

Seven categories numbered from 1 to 7 in the Cover_Type column, to be classified:
1: Spruce/Fir
2: Lodgepole Pine
3: Ponderosa Pine
4: Cottonwood/Willow
5: Aspen
6: Douglas-fir
7: Krummholz

Approach: The classical machine learning tasks like Data Exploration, Data Cleaning,
Feature Engineering, Model Building and Model Testing. Try out different machine
learning algorithms that’s best fit for the above case.

Results: You have to build a solution that should able to predict seven different cover
types in four different wilderness areas of the Roosevelt National Forest of Northern
Colorado with the best accuracy. """

# Liraries for data manipulation and to read the data
import pandas as pd
import numpy as np

# Libraries for visualization
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

# Libraries for preprocessing the datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

# Machine Learning Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

#########################################################################################################################
############################################### Cleaning The Datasets ###################################################
#########################################################################################################################

# datasets view code
pd.set_option("display.max_columns",None)
pd.set_option("display.expand_frame_repr", None)

# To read datsets
fp = pd.read_csv("Forest Cover Type Prediction.csv")
print("\n To print top 10 of the datasets : \n",fp.head(10))

# Information about the datasets
print("\n To print information about the datasets :\n",fp.info())

# describe about datasets
print("\n To print description about the datasets :\n", fp.describe())

# shape of the datasets
print("\n To print the shape of the datasets :\n", fp.shape)

# To know unique values in datasets
print("\n To know unique values in datasets :", fp.nunique())

# To find null values
print("\n To find null values : \n", fp.isnull().sum())

# To find duplicate values in rows and columns
print("\n To find duplicate values : \n",fp.duplicated().sum())

# To find columns in datasets
print("\n To print columns : \n", fp.columns)

# Remove unnecessary columns in the datasets
remove_columns = [col for col in fp.columns if fp[col].std() <= 0.02]
fp= fp.drop(columns=remove_columns)

print("Removed columns from the datasets :",remove_columns)

# To find value counts for the Elevation
fp_value_counts = fp['Cover_Type'].value_counts(ascending=False)
print("To know value_counts : ", fp_value_counts)

# Seperate numerical and categorical columns
num_feat = fp.iloc[:,:10]
print("column present in numerical features :", num_feat.columns)

cat_feat = fp.iloc[:,10:54]
print("\n column present in categorical features :\n", cat_feat.columns)

# change column name in the datasets

fp = fp.rename(columns={"Horizontal_Distance_To_Hydrology" : "Hor_Dist_water",
                                      "Vertical_Distance_To_Hydrology" : "vert_Dist_to_water",
                                      "Horizontal_Distance_To_Roadways" : "Hor_Dist_Roadways",
                                      "Hillshade_9am" : "Hillshade_Morning",
                                      "Hillshade_Noon" : "Hillshade_Afternoon",
                                      "Hillshade_3pm" : "Hillshade_Evening",
                                      "Horizontal_Distance_To_Fire_Points" : "Hor_Dist_F_points",
                                      "Wilderness_Area1" : "Rawah", "Wilderness_Area2" : "Neota",
                                      "Wilderness_Area3" : "Comanche Peak",
                                      "Wilderness_Area4" : "Cache la Poudre",
                                      })

# Below code to view updates columns
updated_colums = fp.columns
print("\n Updated columns : \n", updated_colums)

# Updating names in the column using mapping methods
cover_type_mapping = {
    1 : "Spruce/Fir",
    2 : "Lodgepole Pine",
    3 : "Ponderosa Pine",
    4 : "Cottonwood/Willow",
    5 : "Aspen",
    6 : "Douglas-fir",
    7 : "Krummholz"
}

# apply mapping to the cover_type column
fp["Cover_Type"] = fp["Cover_Type"].map(cover_type_mapping)

fp['Wild Areas'] = (fp.iloc[:,10:15]==1).idxmax(1)
fp['Soil Types'] = (fp.iloc[:,15:55]==1).idxmax(1)

# drop the unwanted columns from the datasets
fp = fp.drop(columns = ['Rawah', 'Neota',
       'Comanche Peak', 'Cache la Poudre', 'Soil_Type1', 'Soil_Type2',
       'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
        'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14',  'Soil_Type16','Soil_Type17',
       'Soil_Type18', 'Soil_Type19', 'Soil_Type20','Soil_Type21',
        'Soil_Type22', 'Soil_Type23', 'Soil_Type24','Soil_Type25',
        'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35','Soil_Type37',
        'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])

# Display the unique element in the wild areas
print("\n print the unique areas in wild areas :\n",fp['Wild Areas'].unique())

# to view changes use unique method
print("\n To view the changes in cover_type :\n ", fp['Cover_Type'].unique())

# print the datasets to view the modified changes
print("updated datasets :", fp.head(3))

# To check the datasets after modification
print("\n To check the shape of the datasets :\n",fp.shape)

# To check desciption and information after modified columns
print("\n To check the description of the datasets \n:",fp.describe())

print("\n To check the information of the datasets \n:", fp.info())

print("\nSoil_type_names : \n", fp['Soil Types'])
########################################################################################################################
# use map method to change the name of the soil with forest datasets mostly used in real world scenario
soil_type_mapping = {'Soil_Type1' : 'Aspen', 'Soil_Type2' : 'Clay','Soil_Type3' : 'Slit', 'Soil_Type4' : 'Sand','Soil_Type5' : 'Loam',
                     'Soil_Type6' : 'Peat', 'Soil_Type7' : 'Saline','Soil_Type8' : 'Chalk', 'Soil_Type9' : 'Podzol','Soil_Type10' : 'Chernozem',
                     'Soil_Type11' : 'Andosol','Soil_Type12' : 'Histosol', 'Soil_Type13' : 'Mollisol','Soil_Type14' : 'Aridisol',
                     'Soil_Type15' : 'Vertisol','Soil_Type16' : 'Spodosol', 'Soil_Type17' : 'Alfisol','Soil_Type18' : 'Ultisol',
                     'Soil_Type19' : 'Oxisol','Soil_Type20' : 'Inceptisol', 'Soil_Type21' : 'Entisol','Soil_Type22' : 'Gleysol',
                     'Soil_Type23' : 'Regosol','Soil_Type24' : 'Luvisol', 'Soil_Type25' : 'Ferralsol','Soil_Type26' : 'Calcisol',
                     'Soil_Type27' : 'Cryosol','Soil_Type28' : 'Xerosol', 'Soil_Type29' : 'Cambisol','Soil_Type30' : 'Phaeozem',
                     'Soil_Type31' : 'Nitisol','Soil_Type32' : 'Acrisol', 'Soil_Type33' : 'Durisol','Soil_Type34' : 'Ferralsol',
                     'Soil_Type35' : 'Lithosol','Soil_Type36' : 'Albeluvisol','Soil_Type37' : 'Arenosol', 'Soil_Type38' : 'Gleysol',
                     'Soil_Type39' : 'Fluvisol', 'Soil_Type40' : 'Vertic'}

fp['Soil Types'] = fp['Soil Types'].map(soil_type_mapping)
print("\n To View the name changed Soil type Column : \n",fp['Soil Types'].unique() )

########################################################################################################################
######################################### Exploratory Data Analysis ####################################################
########################################################################################################################

# Define numerical and categorical features
numerical_features = {feature for feature in fp.columns if fp[feature].dtype !='O'}
categorical_features = {feature for feature in fp.columns if fp[feature].dtype == 'O'}

print("We have {} numerical features : {} ".format(len(numerical_features), numerical_features))
print("We have {} categorical features : {} ".format(len(categorical_features), categorical_features))

# Plot distributions for columns present in numeric features
for i in numerical_features:
    plt.figure(figsize=(8,6))
    sns.histplot(fp[i], kde=True, color='blue')
    plt.title(f'Distributions of {i}')
    plt.tight_layout()
plt.show()
#######################################################################################################################
#Plot Distributions for columns present in categorical features:
#Convert set to list for easier indexing

categorical_features = list(categorical_features)

#Set plot size and layout
plt.figure(figsize=(16, 10))
plt.suptitle("Vertical Bar Plots for Categorical Features", fontsize=16)

# Loop through categorical features
for i in range(len(categorical_features)):
    plt.subplot(2, 2, i + 1)  # Adjust rows/columns based on number of features
    sns.countplot(x=fp[categorical_features[i]])

    # Set title and font sizes
    plt.title(f"Distribution of {categorical_features[i]}", fontsize=10)
    plt.xlabel(categorical_features[i], fontsize=6)
    plt.ylabel("Count", fontsize=6)
    plt.xticks(rotation=45, fontsize=6)  # Rotate x-axis labels for better visibility
    plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.tight_layout()
plt.show()
########################################################################################################################
# check for multicolinearity using heatmap for only numerical features:
numerical_data = fp.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
correlation_matrix = numerical_data.corr()

# Mask to remove redundant information
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Numerical Features")
plt.xticks(rotation=45, fontsize=6)
pl.tight_layout()
plt.show()

########################################################################################################################
# To check multicolinearity for categorical features
chi2_test = []
for feature in categorical_features:
    if chi2_contingency(pd.crosstab(fp['Cover_Type'], fp[feature]))[1] < 0.05:
        chi2_test.append('Reject Null Hypothesis')
    else:
        chi2_test.append('Fail to Reject Null Hypothesis')
result = pd.DataFrame(data=[categorical_features, chi2_test]).T
result.columns=['Column', 'Hypothesis Result']
print(result)

# To check null values
print("\n To print the null values is : \n ",fp.isnull().sum())
########################################################################################################################

# How is the data distribution of our target column 'Cover Type'
plt.figure(figsize=(15,6))
sns.histplot(data=fp,x='Cover_Type', hue='Cover_Type', palette='Set1')
plt.title('Forest Cover Type Distribution')
plt.show()

# Find cover_type present in wild_area
wild_area_cover_type = fp.groupby(['Wild Areas'])['Cover_Type'].value_counts().unstack()

wild_area_cover_type.plot(kind='bar',stacked = True, figsize=(20,10))
plt.show()
print("Cover_type value present in wild area : ",wild_area_cover_type)

# Soil Types available for all the wilderness areas
plt.figure(figsize = (15,10))
sns.histplot(data=fp, x = 'Soil Types', hue='Soil Types', palette='Set1')
plt.title("Forest Soil Types Distribution ")
plt.xticks(rotation =90)
plt.tight_layout()
plt.show()
########################################################################################################################
##creating subplots for all the numerical features and comparing them individually

# subplots for elevation in horizontal to water and roadways
fig, axs = plt.subplots(1,2, figsize=(15,7))
plt.subplot(121)
sns.scatterplot(data=fp, y='Hor_Dist_F_points', x='Elevation', hue='Cover_Type')
plt.legend(fp['Cover_Type'], loc='upper left')
plt.tight_layout()
plt.subplot(122)
sns.scatterplot(data=fp, y='Hor_Dist_Roadways', x='Elevation', hue='Cover_Type')
plt.legend(fp['Cover_Type'], loc='upper left')
plt.tight_layout()
plt.show()

# Subplots for horizontal distance to water and vertical distance to water
fig, axs = plt.subplots(1,2, figsize = (15,7))

# Horizontal distance to water vs elevation
sns.scatterplot(data=fp, y='Hor_Dist_water', x='Elevation', hue = 'Cover_Type', ax=axs[0])
axs[0].set_title('Horizontal Distance to water vs Elevation')
axs[0].legend(loc='upper right')

sns.scatterplot(data=fp, y='vert_Dist_to_water', x='Elevation', hue='Cover_Type', ax= axs[1])
axs[1].set_title('Vertical Distance to water vs Elevation')
axs[1].legend(loc = 'upper right')
plt.tight_layout()
plt.show()
########################################################################################################################

# plot the figure size
plt.figure(figsize=(15,7))

# hori to water from hydrant and vert from water to hydrant
plt.subplot(1,2,1)
sns.scatterplot(data = fp, x = 'Hor_Dist_F_points', y='Hor_Dist_water', hue='Cover_Type')
plt.title('Horizontal Distance to Fire Points vs HD water')
plt.legend(loc = 'upper right')

plt.subplot(1,2,2)
sns.scatterplot(data=fp, y='vert_Dist_to_water', x='Hor_Dist_F_points', hue='Cover_Type')
plt.title('Horizontal Distance to Fire Points vs WD water')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
########################################################################################################################
# Visualize data Aspect and Slope for the cover type:
sns.scatterplot(data=fp, y='Slope', x='Aspect', hue='Cover_Type')
plt.title('Visualizing the Aspect and Slope of the Cover Type')
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

########################################################################################################################
# To plot the graph to identify the outliers
plt.figure(figsize=(12,6))
sns.boxplot(data=fp[['Aspect', 'Hor_Dist_F_points',
                    'Elevation', 'Slope', 'Hor_Dist_water',
                    'Hillshade_Evening', 'vert_Dist_to_water',
                    'Hillshade_Afternoon', 'Hillshade_Morning',
                    'Hor_Dist_Roadways']])
plt.title('Detect Outliers by using BoxPlot in numerical features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
########################################################################################################################

# To find symmetric and skew from the numerical features
plt.figure(figsize=(15,7))
for i, feature in enumerate(numerical_features):
    plt.subplot(2,5,i+1)
    sns.violinplot(data=fp, x=feature, color='orange')
    plt.title(f'{feature}', fontsize=8)
    plt.xlabel('')

plt.suptitle('Box plots to visualize skew and symmetry', fontsize =8)
plt.tight_layout(rect=[0,0,1,0.96])

plt.show()
########################################################################################################################

# plotting the difference in co-relations
plt.figure(figsize=(15,7))
numerical_features = list(numerical_features)
correlation_matrix = fp[numerical_features].corr()

# Filter strong correlations (threshold > 0.5 or < -0.5)
strong_corrs = correlation_matrix[(correlation_matrix > 0.5) | (correlation_matrix < -0.5)]
sns.heatmap(strong_corrs, annot=True, cmap='coolwarm', center=0)
plt.title("Heatmap of Strong Correlations")
plt.show()

#################################################################################################################
########################################### Data Cleaning Process ###############################################
#################################################################################################################

# To view top 10 rows of datasets
print("\nTo view top 10 rows of datasets : \n", fp.head(10))

# Checking for NAN values
fp_with_na = [features for features in fp.columns if fp[features].isnull().sum()>=1]
for feature in fp_with_na:
    print(feature, np.round(fp[feature].isnull().mean()*100, 5), '% missing values')
print("\n To check any nan is present : \n", fp_with_na)


print("To view top 3 row in the datasets :", fp.head(3))

#######################################################################################################################
################################################ Feature Engineering ##################################################
#######################################################################################################################

# Different types of available features for prediction
# Numerical features available in the datasets
numerical_features = [feature for feature in fp.columns if fp[feature].dtype != 'O']
print("Number of numerical features available in the datasets:", len(numerical_features))

# Categorical Features available in the datasets
categorical_features = [feature for feature in fp.columns if fp[feature].dtype == 'O']
print('Number of categorical features available in the datasets :', len(categorical_features))

# Scaling the continuous features in the datasets
scaler = StandardScaler()
scaled_features = scaler.fit_transform(fp[numerical_features])

# Convert the scaled features into a dataframe
scaled_df = pd.DataFrame(scaled_features, columns=numerical_features)
print('\nScaled Features in the DataFrame : \n', scaled_df.head(4))

# Discrete features
discrete_features = [feature for feature in numerical_features if(len(fp[feature].unique()) <= 1000)]
print("Discrete features for feature engineering : ",len(discrete_features))

# Continuous feature
cont_feature = [feature for feature in numerical_features if (len(fp[feature].unique()) > 25)]
print('Continuous feature for feature engineering : ',len(cont_feature))

########################################################################################################################
# check and remove outliers present in the numeric features:
plt.figure(figsize=(15,20))
plt.suptitle('Outliers present in the datasets', fontsize=20, fontweight='bold', alpha=0.8, y=1)
for i, feature in enumerate(cont_feature):
    plt.subplot(4,3,i+1)
    sns.set_style('ticks')
    sns.boxplot(fp[cont_feature[i]])

plt.tight_layout()
plt.show()

########################################################################################################################
def cap_outliers_iqr(df, col):
    # Compute IQR
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    # Define upper and lower bounds
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr

    # Cap the outliers
    df[col] = np.where(df[col] > upper_limit, upper_limit,
                       np.where(df[col] < lower_limit, lower_limit, df[col]))

# Apply the outlier removal method to continuous features
for col in cont_feature:
    cap_outliers_iqr(fp, col)

print("Outliers capped using the IQR method.")

plt.figure(figsize=(15,20))
plt.suptitle('Removed Outliers', fontsize=20, fontweight = 'bold', alpha = 0.8, y=1)
for i in range(0, len(cont_feature)):
    plt.subplot(5,2,i+1)
    sns.set_style('ticks')
    sns.boxplot(fp[cont_feature[i]])
    plt.tight_layout()
plt.show()

########################################################################################################################
# Count the value of Wild Areas:
print(fp['Wild Areas'])
wilderness_areas = sorted(fp['Wild Areas'].value_counts().index.tolist())
print("Value Count of Wild Areas :", wilderness_areas)

########################################################################################################################
# Use One Hot Encoded algorith to the datasets
ohe = OneHotEncoder(sparse_output=False)
# convert categorical columns into 0 and 1
ohe_en = ohe.fit_transform(fp[['Wild Areas']])
print("Ohe -en :", ohe_en)
# get feature names out from the wild areas column.
column_names_encoded_data = ohe.get_feature_names_out(['Wild Areas'])
print("column_names_encoded_data :",column_names_encoded_data)

# converting into dataframe combining ohe_en and columns_name_encoded_data
new_df = pd.DataFrame(ohe_en, columns= column_names_encoded_data)
print("new_df :", new_df)

# Removing the wild areas from the column
fp.drop(columns=['Wild Areas'], inplace=True)

fp = pd.concat([fp,new_df], axis=1)
print("Combining old datasets to new_datasets :", fp.head(3))
########################################################################################################################
# Rename the columns of the wilderness area
fp = fp.rename(columns = {
    "Wild Areas_Cache la Poudre" : "Cache la Poudre", "Wild Areas_Comanche Peak" : "Comanche Peak",
    "Wild Areas_Neota" : "Neota", "Wild Areas_Rawah" : "Rawah"
})
########################################################################################################################
# Preprocessing "Soil Types " using Label Encoder to do model test
label_obj = LabelEncoder()
fp[('Soil Types')] = label_obj.fit_transform(fp[('Soil Types')])
print("updated Soil Types of head :",fp['Soil Types'])

# Display the first few rows
print("Top 3 Rows to be printed:\n", fp.head(3))

feature_names = cont_feature + wilderness_areas + ['Soil Types'] + ['Cover_Type']
print("Combined Features :", feature_names)
all_features = fp[feature_names]

# Information about the Soil_type column for better performance
print(all_features.info())
print("Soil types information :\n ", fp.head())

# To count the value of cover_type
print("The first five rows of cover_type :",all_features['Cover_Type'].head())
print("Count the values of cover_type :",all_features['Cover_Type'].value_counts())
print("All features null count :", all_features.isnull())

# plot a pie chart for visualize the cover_type columns
cover_type_counts = all_features['Cover_Type'].value_counts()

# Visualize the cover type after combined all features:
plt.figure(figsize=(15,7))
sns.barplot(x=cover_type_counts.index, y=cover_type_counts.values,color='green', hue=None)
plt.title('Distribution of Cover_Type', fontsize=16, fontweight = 'bold', style='italic')
plt.xlabel('Cover_Type', fontsize = 12, style='italic')
plt.ylabel('Count', fontsize=12, style='italic')
plt.show()

########################################################################################################################
################################## Train and Test for the preprocessed datasets ########################################
########################################################################################################################
def split_data(fp):
    # define feature and target for perform the prediction
    x = fp.drop('Cover_Type', axis=1)
    y = fp['Cover_Type']

    # train and split the processed datasets
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    # scale data using Standard Scaler
    scaler = StandardScaler()
    scaler.fit_transform(x_train)

    x_train = pd.DataFrame(scaler.transform(x_train),columns=x_train.columns)
    x_test  = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    return x_train,x_test,y_train,y_test

########################################################################################################################
def evaluate_model(model_RF, x_test,y_test):
    model_accuracy = model_RF.score(x_test, y_test)
    print(f" Model Accuracy:{model_accuracy:.2%}")

    y_predict = model_RF.predict(x_test)

    cm = confusion_matrix(y_test,y_predict)
    c1_report = classification_report(y_test,y_predict,zero_division=1)

    # ploting the visualization for confusion matrix
    plt.figure(figsize=(15,7))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion matrix")
    plt.show()

    print("Classification report", c1_report)
    return model_accuracy

########################################################################################################################
############################################ Randomforest Classifier ###################################################
########################################################################################################################
modified_data = all_features.copy()
x_train,x_test,y_train,y_test = split_data(modified_data)

model_rf =RandomForestClassifier()
model_rf.fit(x_train,y_train)

rf_accuracy =evaluate_model(model_rf, x_test,y_test)
print("Evaluated RandomClassifier Model :",rf_accuracy )

########################################################################################################################
######################################### Predict using Logistic Regression ############################################
########################################################################################################################
# Train the Logistic Regression model
model_lr = LogisticRegression(max_iter=100, class_weight='balanced', solver='saga')
model_lr.fit(x_train, y_train)

lr_accuracy = evaluate_model(model_lr,x_test, y_test)
print("Logistic Regression :", lr_accuracy)

y_pred_probabilities = model_lr.predict_proba(x_test)
print("Y_Pred_Probabilities :",y_pred_probabilities)

########################################################################################################################
# Random Under-Sampling
under_sample = all_features.copy()
print("Under sample value count :",under_sample['Cover_Type'].value_counts())

minimum_class = np.min(under_sample["Cover_Type"].value_counts().values)
print("Size of smallest class :",minimum_class)

class_subsets = [under_sample.query("Cover_Type ==" + str(i)) for i in range(7)]
print("Cover_type 0th index :",class_subsets[0])

# Map Cover_Type strings to integers
cover_type_mapping = {
    'Aspen': 0,
    'Lodgepole Pine': 1,
    'Spruce/Fir': 2,
    'Krummholz': 3,
    'Ponderosa Pine': 4,
    'Douglas-fir': 5,
    'Cottonwood/Willow': 6
}

# Check the unique values in the 'Cover_Type' column
print("Unique values in 'Cover_Type':", under_sample['Cover_Type'].unique())

# Find the minimum class size in the dataset
minimum_class = min(under_sample['Cover_Type'].value_counts())
print(f"Minimum class size: {minimum_class}")

# Create subsets for each class and sample
class_subsets = []
for cover_type, idx in cover_type_mapping.items():
    # Query the rows for each class
    class_subset = under_sample.query(f"Cover_Type == '{cover_type}'")
    # Check if the class has rows before sampling
    if class_subset.shape[0] > 0:
        # Sample the class
        class_subsets.append(class_subset.sample(min(minimum_class, class_subset.shape[0])))
    else:
        print(f"Class {cover_type} has no rows!")

# Concatenate the resampled subsets if they exist
if class_subsets:
    under_sample = pd.concat(class_subsets, axis=0).sample(frac=1.0).reset_index(drop=True)
    print("Under-sampled data count: ", under_sample['Cover_Type'].value_counts())
else:
    print("No valid class data available for undersampling.")

x_train, x_test, y_train, y_test = split_data(under_sample)

model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)

print("Evaluating model for under sample using random classifier :\n",evaluate_model(model_rf,x_test, y_test))

########################################################################################################################
over_sample = all_features.copy()

# Verify the unique values and data type of `Cover_Type`
print("Unique values in Cover_Type:", over_sample["Cover_Type"].unique())
print("Data type of Cover_Type:", over_sample["Cover_Type"].dtype)

# Determine the maximum class size
max_class = over_sample["Cover_Type"].value_counts().max()
print("Size of the largest class (max_class):", max_class)

# Oversampling the minority classes
class_subsets = []

# Handle numeric or string-based `Cover_Type`
for i in over_sample["Cover_Type"].unique():
    # Handle numeric or string-based `Cover_Type`
    if over_sample["Cover_Type"].dtype == 'object':  # If Cover_Type is a string
        subset = over_sample.query(f"Cover_Type == '{i}'")  # Enclose in quotes
    else:  # If Cover_Type is numeric
        subset = over_sample.query(f"Cover_Type == {i}")  # No quotes needed

    # Print the subset shape for debugging
    print(f"Class {i} subset shape: {subset.shape}")

    # Check if the subset is not empty
    if not subset.empty:
        subset = subset.sample(max_class, replace=True, random_state=42)

    # Append the oversampled subset
    class_subsets.append(subset)

# Concatenate all subsets, shuffle, and reset the index
over_sample = pd.concat(class_subsets, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

# Verify the new class distribution
print("New class distribution after oversampling:")
print(over_sample["Cover_Type"].value_counts())

x_train, x_test, y_train, y_test = split_data(over_sample)

# Train the model
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)

# Evaluate the model
print("Over sampling evaluation:", evaluate_model(model_rf, x_test, y_test))

########################################################################################################################
# Randomforest Classifier to calculate roc_auc_score in multi class classification
x_train, x_test, y_train, y_test = split_data(modified_data)

# Train the model
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)


y_pred_prob = model_rf.predict_proba(x_test)
rf_roc_auc = roc_auc_score(y_test, y_pred_prob,multi_class='ovr')
print("roc_score :", rf_roc_auc)
rf_accuracy = evaluate_model(model_rf,x_test, y_test)
print("Randomforest Classifier Accuracy :",rf_accuracy)

########################################################################################################################
# Extra Trees (Random Forest) Classifier
model_exr = ExtraTreesClassifier()
model_exr.fit(x_train,y_train)

y_pre_pro = model_exr.predict_proba(x_test)
exr_roc = roc_auc_score(y_test,y_pre_pro,multi_class='ovr')
print("roc_score", exr_roc)
exr_accuracy = evaluate_model(model_exr,x_test, y_test)
print("Extra_Tree_Classifier_Accuracy :",exr_accuracy)

########################################################################################################################
# Replace whitespace in feature names
x_train.columns = x_train.columns.str.replace(' ', '_')
x_test.columns = x_test.columns.str.replace(' ', '_')

# Light Gradient Boosting Machine(LightGBM) Classifier
model_lgbm = LGBMClassifier(class_weight="balanced", random_state=42)
model_lgbm.fit(x_train,y_train)

y_pre_proba = model_lgbm.predict_proba(x_test)
roc_lgbm = roc_auc_score(y_test,y_pre_proba,multi_class='ovr')
print("LGBM roc score :",roc_lgbm)
model_lgbm_accuracy = evaluate_model(model_lgbm,x_test, y_test)
print("LGBM Accuracy :", model_lgbm_accuracy)

########################################################################################################################

# Encode the labels into integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# XGBClassifier predictions:
model_xgb = XGBClassifier(verbose=False)
model_xgb.fit(x_train, y_train_encoded)

y_p_prob = model_xgb.predict_proba(x_test)
xgb_roc = roc_auc_score(y_test_encoded, y_p_prob, multi_class='ovr')
print("roc_score:", xgb_roc)

# Evaluate the model's accuracy
xgb_accuracy = evaluate_model(model_xgb, x_test, y_test_encoded)
print("XGB accuracy score:", xgb_accuracy)

########################################################################################################################

# model training using CatBoostClassifier
model_cat = CatBoostClassifier()
model_cat.fit(x_train,y_train)

y_pred_probi = model_cat.predict_proba(x_test)
cat_roc = roc_auc_score(y_test, y_pred_probi, multi_class='ovr')
print("cat_roc :",cat_roc)

cat_accuracy = evaluate_model(model_cat,x_test, y_test)
print("CatBoostClassifier Accuracy :",cat_accuracy)

########################################################################################################################
# Model Training for MLPClassifier

model_mlp = MLPClassifier()
model_mlp.fit(x_train,y_train)

y_pred_probhan = model_mlp.predict_proba(x_test)
mlp_roc = roc_auc_score(y_test,y_pred_probhan,multi_class='ovr')
print("mlp_roc",mlp_roc)

mlp_accuracy = evaluate_model(model_mlp,x_test,y_test)
print("MLPClassifier Accuracy :",mlp_accuracy)

########################################################################################################################
# Evaluate all base models
compare_all_models = pd.DataFrame({"Model":["Randomforest_classifier", "Extra_Tree_classifier", "Cat_Boost_Classifier","XGB_Classifier",
                                   "Light_Gradient_Boost_Classifier","Logistic_Regression","MLPClassifier"],
                                   "Accuracy":["rf_accuracy","exr_accuracy","cat_accuracy","xgb_accuracy","model_lgbm_accuracy","lr_accuracy","mlp_accuracy"]})

compare_all_models = compare_all_models.sort_values(by = "Accuracy", ascending=True")

# set the plot to compare all the models
plt.figure(figsize(15,7))
sns.barplot(x="Accuracy", y="Model", data="compare_all_models", palette="magma")
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Accuracy score for all the models :", fontsize=14)
plt.show()
########################################################################################################################


