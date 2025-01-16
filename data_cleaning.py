### IMPORTS & SETUP
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import RocCurveDisplay

import dataframe_image as dfi

seed = 12354635
np.random.seed(seed)

### DATA IMPORT

num_cols = ["avg_rating", "avg_difficulty", "n_ratings", "pepper", "would_take_again_pct", "online_ratings",  "male", "female"]
num = pd.read_csv("rmpCapstoneNum.csv", header=None, names=num_cols)

qualitative_cols = ["major_or_field", "university", "state"]
qualitative = pd.read_csv("rmpCapstoneQual.csv", header=None, names=qualitative_cols)

tags_cols = [
    "tough_grader",
    "good_feedback",
    "respected",
    "lots_to_read",
    "participation",
    "dont_skip_class_or_you_will_not_pass",
    "lots_of_homework",
    "inspirational",
    "pop_quizzes",
    "accessible",
    "so_many_papers",
    "clear_grading",
    "hilarious",
    "test_heavy",
    "graded_by_few_things",
    "amazing_lecturer",
    "caring",
    "extra_credit",
    "group_projects",
    "lecture_heavy",
]
tags = pd.read_csv("rmpCapstoneTags.csv", header=None, names=tags_cols)

### DATA CLEANING

df = num.join(tags)

# drop all data where average rating is null since this data is not useful
# and most other columns are null in these cases
# we also want at least one rating per professor to make meaningful inferences
# this corresponds to dropping ~25% of not-null ratings data
df = df[(df.avg_rating.notnull()) & (df.n_ratings > 1)]
normalize_cols = tags_cols + ["online_ratings"] # cols to normalize by number of ratings

# normalize tag cols to number of ratings
# interpretation becomes "percent of ratings with tag x"
df[normalize_cols] = df[normalize_cols].div(df["n_ratings"], axis=0)

# null_counts = df.isnull().sum()
# would take again percent is still mostly null
# fill with column mean8jmn  vbvb8
mean = df["would_take_again_pct"].mean()
df["would_take_again_pct"].fillna(mean, inplace=True)

# df[(df.male == 1) & (df.female == 1)].shape[0] / df.shape[0] # about 3% of data has both male and female marked as one
# drop this data
df = df[~((df.male == 1) & (df.female == 1))]
