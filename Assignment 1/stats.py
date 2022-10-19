
from mlxtend.evaluate import proportion_difference


# Run the test
print("Proportions Z-Test")

z, p = proportion_difference(0.5, 0.69, n_1=661)
print(f"Random vs tree: z statistic: {z}, p-value: {p}\n")

z, p = proportion_difference(0.5, 0.78, n_1=661)
print(f"Random vs Bagging: z statistic: {z}, p-value: {p}\n")

z, p = proportion_difference(0.5, 0.76, n_1=661)
print(f"Random vs RF: z statistic: {z}, p-value: {p}\n")


z, p = proportion_difference(0.69, 0.78, n_1=661)
print(f"Tree vs Bagging: z statistic: {z}, p-value: {p}\n")

z, p = proportion_difference(0.69, 0.76, n_1=661)
print(f"Tree vs RF: z statistic: {z}, p-value: {p}\n")

z, p = proportion_difference(0.78, 0.76, n_1=661)
print(f"Baggging vs RFz statistic: {z}, p-value: {p}\n")
