import torch
import faiss
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ==== CONFIG ====
MODEL_NAME = "microsoft/codebert-base"
DATA_FILE = "parent_code_comment_with_title-final.csv"
EMB_FILE = "code_embeddings.npy"
INDEX_FILE = "faiss_index.bin"
MAP_FILE = "code_map.npy"
BATCH_SIZE = 64
MAX_LEN = 256
# =================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
model.half().eval()  # use FP16 for speed

# Load dataset
df = pd.read_csv(DATA_FILE, dtype=str)
print(f"Loaded dataset with {len(df)} rows.")

# Collect all code snippets
codes, code_map = [], []
for col in ["QuestionCode", "AnswerCode"]:
    if col in df.columns:
        for idx, val in df[col].dropna().items():
            codes.append(str(val))
            code_map.append((idx, col))
print(f"Collected {len(codes)} code snippets.")

# Function for batch embeddings
def get_embeddings_batch(codes, batch_size=BATCH_SIZE):
    all_embs = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
        # Convert inputs to half precision
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embs.append(embs)
        if i % 6400 == 0:
            print(f"Processed {i}/{len(codes)}")
    return np.vstack(all_embs)

# Generate and save embeddings if not already saved
try:
    embeddings = np.load(EMB_FILE)
    print(f"Loaded precomputed embeddings: {embeddings.shape}")
except FileNotFoundError:
    print("Computing embeddings...")
    embeddings = get_embeddings_batch(codes)
    np.save(EMB_FILE, embeddings)
    np.save(MAP_FILE, np.array(code_map, dtype=object))
    print(f"Saved embeddings to {EMB_FILE}")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, INDEX_FILE)
print(f"FAISS index saved to {INDEX_FILE}")
