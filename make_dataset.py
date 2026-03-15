from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["cancer_type"] = data.target

# Save dataset
df.to_csv("cancer_gene_expression.csv", index=False)

print("Dataset created!")
