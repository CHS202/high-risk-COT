# prompt: combine same category of race in feature_gender_df, e.g. ASIAN - ASIAN INDIAN and ASIAN - KOREAN should all be ASIAN

# Assuming 'feature_gender_df' is your DataFrame and the race information is in a column named 'race'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score, f1_score, recall_score

def combine_race_categories(df):
  """Combines similar race categories in a DataFrame.

  Args:
      df: The input DataFrame with a 'race' column.

  Returns:
      A DataFrame with combined race categories.
  """

  df['race'] = df['race'].astype(str) # Ensure the column is treated as string
  df['race'] = df['race'].str.upper() # Convert to uppercase to handle inconsistencies

  # Define a mapping for race categories
  race_mapping = {
      'AMERICAN INDIAN/ALASKA NATIVE: ': 'AMERICAN INDIAN',
      'ASIAN': 'ASIAN',
      'ASIAN - ASIAN INDIAN': 'ASIAN',
      'ASIAN - KOREAN': 'ASIAN',
      'ASIAN - CHINESE': 'ASIAN',
      'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
      'BLACK/AFRICAN': 'BLACK',
      'BLACK/AFRICAN AMERICAN': 'BLACK',
      'BLACK/CAPE VERDEAN': 'BLACK',
      'BLACK/CARIBBEAN ISLAND': 'BLACK',
      'HISPANIC OR LATINO': 'HISPANIC',
      'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
      'HISPANIC/LATINO - DOMINICAN': 'HISPANIC',
      'HISPANIC/LATINO - SALVADORAN': 'HISPANIC',
      'HISPANIC/LATINO - CENTRAL AMERICAN': 'HISPANIC',
      'HISPANIC/LATINO - MEXICAN': 'HISPANIC',
      'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC',
      'HISPANIC/LATINO - CUBAN': 'HISPANIC',
      'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC',
      'HISPANIC/LATINO - HONDURAN': 'HISPANIC',
      'MULTIPLE RACE/ETHNICITY': 'MULTIPLE',
      'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
      'OTHER': 'OTHER',
      'PATIENT DECLINED TO ANSWER': 'UNKNOWN',
      'PORTUGUESE': 'PORTUGUESE',
      'SOUTH AMERICAN': 'SOUTH AMERICAN',
      'UNABLE TO OBTAIN': 'UNKNOWN',
      'UNKNOWN': 'UNKNOWN',
      'WHITE': 'WHITE',
      'WHITE - BRAZILIAN': 'WHITE',
      'WHITE - RUSSIAN': 'WHITE',
      'WHITE - OTHER EUROPEAN': 'WHITE',
      'WHITE - EASTERN EUROPEAN': 'WHITE'
  }

  # Replace values using the mapping
  for old_value, new_value in race_mapping.items():
      df.loc[df['race'].str.contains(old_value, na=False, case=False), 'race'] = new_value

  return df

TARGET_VARIABLE = 'outcome'  # The column name of your target variable
# Example usage:
# Assuming your dataframe is named feature_gender_df
feature_gender_df = pd.read_csv('data/raw/gender-race.csv')
feature_gender_df = combine_race_categories(feature_gender_df)

feature_df = pd.read_csv('data/raw/feature_extracted_all.csv')
features_df = pd.merge(feature_df, feature_gender_df, on='stay_id', how='outer')

X = features_df.drop(TARGET_VARIABLE, axis=1)
y = features_df[TARGET_VARIABLE]

print("gender bias check")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_test = np.loadtxt('data/model_multi/NN/y_test_NN.csv', delimiter=',')

# Debug: Check types
print(f"Type of X_test: {type(X_test)}")
print(f"Type of y_test: {type(y_test)}") # This will confirm if y_test is a NumPy array

# M_indices and F_indices are actual index labels from X_test (if X_test is a DataFrame)
# These are useful if you need to refer back to the original DataFrame.
M_original_indices = X_test[X_test['gender'] == 'M'].index
F_original_indices = X_test[X_test['gender'] == 'F'].index

# If X_test is a DataFrame (it should be), you can create X_test_M/F using .loc
X_test_M = X_test.loc[M_original_indices]
X_test_F = X_test.loc[F_original_indices]

# Create boolean masks from X_test. These are positionally aligned with X_test's rows.
# .values converts the boolean pandas Series to a NumPy boolean array.
is_male_in_X_test_mask = (X_test['gender'] == 'M').values
is_female_in_X_test_mask = (X_test['gender'] == 'F').values

# If y_test is a NumPy array, use the boolean mask for subsetting:
if isinstance(y_test, np.ndarray):
    y_test_M = y_test[is_male_in_X_test_mask]
    y_test_F = y_test[is_female_in_X_test_mask]
elif isinstance(y_test, pd.Series): # If y_test is a Series (expected behavior)
    y_test_M = y_test.loc[M_original_indices]
    y_test_F = y_test.loc[F_original_indices]
else:
    raise TypeError(f"y_test is of unexpected type: {type(y_test)}")

print(len(y_test_M), len(y_test_F))

# load y_pred_proba_NN.csv
y_pred_proba = np.loadtxt('data/model_multi/NN/y_pred_proba_NN.csv', delimiter=',')
y_pred_classes = (y_pred_proba > 0.5).astype(int)  # Convert to class labels
# Filter the NumPy array y_pred_proba_for_X_test using the boolean masks
y_pred_proba_of_males_in_test = y_pred_proba[is_male_in_X_test_mask]
y_pred_proba_of_females_in_test = y_pred_proba[is_female_in_X_test_mask]
y_pred_classes_of_males_in_test = y_pred_classes[is_male_in_X_test_mask]
y_pred_classes_of_females_in_test = y_pred_classes[is_female_in_X_test_mask]

# calculate AUC for M and F
auc_M = roc_auc_score(y_test_M, y_pred_proba_of_males_in_test)
auc_F = roc_auc_score(y_test_F, y_pred_proba_of_females_in_test)

print("AUC for M:", auc_M)
print("AUC for F:", auc_F)

# calculate F1 score for M and F
f1_M = f1_score(y_test_M, y_pred_classes_of_males_in_test)
f1_F = f1_score(y_test_F, y_pred_classes_of_females_in_test)

print("F1 score for M:", f1_M)
print("F1 score for F:", f1_F)

# calculate sensitivity for M and F
sensitivity_M = recall_score(y_test_M, y_pred_classes_of_males_in_test)
sensitivity_F = recall_score(y_test_F, y_pred_classes_of_females_in_test)

print("Sensitivity for M:", sensitivity_M)
print("Sensitivity for F:", sensitivity_F)

# calculate specificity for M and F
specificity_M = recall_score(y_test_M, y_pred_classes_of_males_in_test, pos_label=0)
specificity_F = recall_score(y_test_F, y_pred_classes_of_females_in_test, pos_label=0)

print("Specificity for M:", specificity_M)
print("Specificity for F:", specificity_F)

# calculate Youden's Index for M and F
youden_index_M = sensitivity_M + specificity_M - 1
youden_index_F = sensitivity_F + specificity_F - 1

print("Youden's Index for M:", youden_index_M)
print("Youden's Index for F:", youden_index_F)

print("race bias check")
# get indices of race categories from X_test
race_categories = X_test['race'].unique()
print(race_categories)
# Define main race categories
main_races = ['WHITE', 'BLACK', 'ASIAN']

# Create a dictionary to store indices for each race
race_indices = {}
for race in main_races:
    if race in race_categories:
        race_indices[race] = (X_test['race'] == race).values

# Create 'Others' category for races not in main_races
others_mask = ~X_test['race'].isin(main_races)
race_indices['Others'] = others_mask.values

print(race_indices.keys())

# calculate AUC for each race category
for race, indices in race_indices.items():
    y_test_race = y_test[indices]
    print(race, len(y_test_race))
    y_pred_proba_of_race_in_test = y_pred_proba[indices]
    # print(y_pred_proba_of_race_in_test)
    auc_race = roc_auc_score(y_test_race, y_pred_proba_of_race_in_test)
    print(f"AUC for {race}: {auc_race}")

# calculate F1 score for each race category
for race, indices in race_indices.items():
    y_test_race = y_test[indices]
    y_pred_classes_of_race_in_test = y_pred_classes[indices]
    f1_race = f1_score(y_test_race, y_pred_classes_of_race_in_test)
    print(f"F1 score for {race}: {f1_race}")

# calculate sensitivity for each race category
sensitivity_race = {}
for race, indices in race_indices.items():
    y_test_race = y_test[indices]
    y_pred_classes_of_race_in_test = y_pred_classes[indices]
    sensitivity_race[race] = recall_score(y_test_race, y_pred_classes_of_race_in_test)
    print(f"Sensitivity for {race}: {sensitivity_race[race]}")

# calculate specificity for each race category
specificity_race = {}
for race, indices in race_indices.items():
    y_test_race = y_test[indices]
    y_pred_classes_of_race_in_test = y_pred_classes[indices]
    specificity_race[race] = recall_score(y_test_race, y_pred_classes_of_race_in_test, pos_label=0)
    print(f"Specificity for {race}: {specificity_race[race]}")

# calculate Youden's Index for each race category
for race, indices in race_indices.items():
    y_test_race = y_test[indices]
    y_pred_classes_of_race_in_test = y_pred_classes[indices]
    youden_index_race = sensitivity_race[race] + specificity_race[race] - 1
    print(f"Youden's Index for {race}: {youden_index_race}")