# Define the permutation test function
def permutation_test_variance(group1, group2, n_permutations=10000, seed=None):
    if seed:
        np.random.seed(seed)
    # Observed difference in variances
    observed_diff = np.var(group1, ddof=1) - np.var(group2, ddof=1)
    # Combine data
    combined = np.concatenate([group1, group2])
    # Permutation
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:len(group1)]
        perm_group2 = combined[len(group1):]
        perm_diffs.append(np.var(perm_group1, ddof=1) - np.var(perm_group2, ddof=1))
    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value, perm_diffs

# Perform the permutation test
observed_diff, perm_p_value, perm_diffs = permutation_test_variance(male_ratings, female_ratings, seed=12354635)

# Results
print(f"Observed Difference in Variance (Male - Female): {observed_diff:.3f}")
print(f"P-Value: {perm_p_value:.5f}")
print(f"Variance (Male): {np.var(male_ratings, ddof=1):.3f}")
print(f"Variance (Female): {np.var(female_ratings, ddof=1):.3f}")

# Visualization
plt.figure(figsize=(10, 6))
sns.histplot(perm_diffs, kde=True, color="skyblue", bins=30, label="Permutation Distribution")
plt.axvline(observed_diff, color="red", linestyle="--", label=f"Observed Difference: {observed_diff:.3f}")
plt.title("Permutation Test for Variance Difference")
plt.xlabel("Difference in Variance (Male - Female)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
