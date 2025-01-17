# Add a 'gender' column for visualization purposes
df["gender"] = df.apply(lambda x: "Male" if x["male"] == 1 else ("Female" if x["female"] == 1 else "Unknown"), axis=1)

# Group ratings by gender
male_ratings = df[df["male"] == 1]["avg_rating"]
female_ratings = df[df["female"] == 1]["avg_rating"]

# Perform independent t-test
t_stat, p_value = ttest_ind(male_ratings, female_ratings, equal_var=False)

# Print results
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

mean_male = male_ratings.mean()
mean_female = female_ratings.mean()
std_male = male_ratings.std(ddof=1)
std_female = female_ratings.std(ddof=1)
n_male = len(male_ratings)
n_female = len(female_ratings)
var_male = male_ratings.var(ddof=1)
var_female = female_ratings.var(ddof=1)

# Compute pooled standard deviation
pooled_std = np.sqrt(
    ((n_male - 1) * std_male**2 + (n_female - 1) * std_female**2) / (n_male + n_female - 2)
)

# Compute Cohen's d
cohen_d = (mean_male - mean_female) / pooled_std

print(f"Cohen's d: {cohen_d:.3f}")

# Visualizations
# Boxplot of ratings by gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="gender", y="avg_rating", palette="pastel")
plt.title("Distribution of Average Ratings by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# Histogram of ratings by gender
plt.figure(figsize=(8, 6))
sns.histplot(male_ratings, kde=True, color="blue", label="Male", bins=30, alpha=0.6)
sns.histplot(female_ratings, kde=True, color="pink", label="Female", bins=30, alpha=0.6)
plt.title("Histogram of Ratings by Gender")
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
