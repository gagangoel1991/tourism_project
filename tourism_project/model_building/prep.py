# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("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"))
DATASET_PATH = "hf://datasets/ggoel1991/tourism_project/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

#Correcting the Data Anamolies
# Fix Gender column
df['Gender'] = df['Gender'].replace({
    'Fe Male': 'Female'
})

# Fix MaritalStatus column
df['MaritalStatus'] = df['MaritalStatus'].replace({
    'Unmarried': 'Single'
})


#Converting Age, Duration and Incoe to Categorical Values
df['Age_cat'] = pd.cut(df['Age'],
                       bins=[0, 18, 30, 45, 60, 100],
                       labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])

# Convert DurationOfPitch into categories
df['Duration_cat'] = pd.cut(df['DurationOfPitch'],
                            bins=[0, 5, 15, 30, 60, df['DurationOfPitch'].max()],
                            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])

# Convert MonthlyIncome into categories
df['Income_cat'] = pd.cut(df['MonthlyIncome'],
                          bins=[0, 10000, 20000, 40000, 70000, df['MonthlyIncome'].max()],
                          labels=['Low Less than 10K', 'Lower-Middle Greator than 10K  and Less than 20K', 'Middle Greator than 20K  and Less than 40K', 'Upper-Middle Greator than 40K  and Less than 70K', 'High Greator than 70K'])


# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID','Age','DurationOfPitch','MonthlyIncome'], inplace=True)



# Encode categorical columns
label_encoder = LabelEncoder()
df['Age_cat'] = label_encoder.fit_transform(df['Age_cat'])
df['Duration_cat'] = label_encoder.fit_transform(df['Duration_cat'])
df['Income_cat'] = label_encoder.fit_transform(df['Income_cat'])
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])
df['CityTier'] = label_encoder.fit_transform(df['CityTier'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoder.fit_transform(df['Designation'])
df['PreferredPropertyStar']=label_encoder.fit_transform(df['PreferredPropertyStar'])
df['OwnCar']=label_encoder.fit_transform(df['OwnCar'])
df['PitchSatisfactionScore']=label_encoder.fit_transform(df['PitchSatisfactionScore'])
df['Passport']=label_encoder.fit_transform(df['Passport'])

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="ggoel1991/tourism_project",
        repo_type="dataset",
    )
