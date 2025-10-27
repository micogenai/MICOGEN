import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import ollama

#  CONFIG 
DATA_FILE = "parent_code_comment_with_title-final.csv"
EMB_FILE = "code_embeddings.npy"
INDEX_FILE = "faiss_index.bin"
MAP_FILE = "code_map.npy"
MODEL_NAME = "microsoft/codebert-base"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load components
print("Loading FAISS index and embeddings...")
index = faiss.read_index(INDEX_FILE)
embeddings = np.load(EMB_FILE)
code_map = np.load(MAP_FILE, allow_pickle=True)
df = pd.read_csv(DATA_FILE, dtype=str)

# Reload CodeBERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
model.eval()

def get_embedding(code: str):
    if not isinstance(code, str) or code.strip() == "":
        return np.zeros(768)
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

#  INPUT EXAMPLE 
input_code = """"public void addColumn(String header){
  WTableColumn tableColumn;
  tableColumn=new WTableColumn();
  tableColumn.setHeaderValue(Util.cleanAmp(header));
  setColumnVisibility(tableColumn,true);
  m_tableColumns.add(tableColumn);
  return;
}
"
"""

#  FAISS SEARCH 
input_emb = np.expand_dims(get_embedding(input_code), axis=0)
k = 50
D, I = index.search(input_emb, k)
similarities = 1 / (1 + D[0])

#  FILTER AND BUILD CONTEXT 
filtered = [(i, sim) for i, sim in zip(I[0], similarities) if sim >= 0.6]
filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:3]

context_parts = []
for i, sim in filtered:
    row_idx, col_type = code_map[i]
    row = df.iloc[row_idx]

    title = str(row.get("Title", "")).strip()
    qcode = str(row.get("QuestionCode", "")).strip()
    qcomment = str(row.get("QuestionComment", "")).strip()
    acode = str(row.get("AnswerCode", "")).strip()
    acomment = str(row.get("AnswerComment", "")).strip()
    qintent = str(row.get("QuestionIntent", "")).strip()
    aintent = str(row.get("AnswerIntent", "")).strip()

    context_parts.append(f"""
Title: {title}
Similarity: {sim:.2f}

QuestionIntent: {qintent}
AnswerIntent: {aintent}

QuestionCode:
{qcode[:300]}

QuestionComment:
{qcomment}

AnswerCode:
{acode[:300]}

AnswerComment:
{acomment}
""")

context = "\n\n---\n\n".join(context_parts)

#  BUILD PROMPT 
prompt = f"""
Generate a one-line meaningful comment for the following code:

{input_code}

Use the following similar examples (code, comments, and intents) as context:

{context}
"""

#  GENERATE COMMENT 
try:
    response = ollama.chat(
        model="codellama:7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    print("\nGenerated Comment:\n", response["message"]["content"])
except Exception as e:
    print(f"Ollama model error: {e}")
