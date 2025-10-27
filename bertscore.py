import pandas as pd
from bert_score import score

# Load your CSV file (replace path with yours)
df = pd.read_csv("dataset-with-generated-comments.csv")

# Columns: 'comment' = reference, 'GeneratedComment' = generated
references = df['comment'].astype(str).tolist()
candidates = df['GeneratedComment'].astype(str).tolist()

# Compute BERTScore (P=Precision, R=Recall, F1=Final Score)
P, R, F1 = score(candidates, references, lang="en", verbose=True)

# Add the mean F1 score for each pair to the dataframe
df['BERTScore'] = F1.tolist()
df['Precision']=P.tolist()
df['Recall']=R.tolist()
# Save results to a new CSV
df.to_csv("model-bertscore.csv", index=False)

print("BERTScore calculation complete! Saved as model-bertscore.csv")
print(df[['comment', 'GeneratedComment', 'BERTScore']].head())



