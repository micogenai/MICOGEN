import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm
import nltk
import openai


#  INITIAL SETUP

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==== CONFIG ====
DATA_FILE = "manually_labeled_data_20000.xlsx"
OUTPUT_FILE = "gpt4omini_goal_dataset_with_metrics.csv"
MODEL_NAME = "microsoft/codebert-base"
EMB_FILE = "code_embeddings.npy"
INDEX_FILE = "faiss_index.bin"
MAP_FILE = "code_map.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 50
SIM_THRESHOLD = 0.6
NUM_SAMPLES = 150
OPENAI_API_KEY = "YOUR_API_KEY"

device = torch.device(DEVICE)
print(f"Using device: {device}")


#  LOAD MODELS AND DATA

openai.api_key = OPENAI_API_KEY

index = faiss.read_index(INDEX_FILE)
embeddings = np.load(EMB_FILE)
code_map = np.load(MAP_FILE, allow_pickle=True)

df_original = pd.read_csv("parent_code_comment_with_title-final.csv", dtype=str)
goal_df = pd.read_excel(DATA_FILE, dtype=str).head(NUM_SAMPLES)
goal_df["GeneratedComment"] = ""
goal_df["BLEU"] = 0.0
goal_df["ROUGE_L"] = 0.0
goal_df["METEOR"] = 0.0

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
model.eval()


#  FUNCTIONS

def get_embedding(code: str):
    if not isinstance(code, str) or code.strip() == "":
        return np.zeros(768)
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

def generate_comment(input_code: str, input_intent: str = "") -> str:
    """Use FAISS + GPT-4o Mini to generate a one-line comment"""
    try:
        input_emb = np.expand_dims(get_embedding(input_code), axis=0)
        D, I = index.search(input_emb, TOP_K)
        similarities = 1 / (1 + D[0])

        filtered = [(i, sim) for i, sim in zip(I[0], similarities) if sim >= SIM_THRESHOLD]
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:3]

        context_parts = []
        for i, sim in filtered:
            row_idx, col_type = code_map[i]
            row = df_original.iloc[row_idx]

            qintent = str(row.get("QuestionIntent", "")).strip()
            aintent = str(row.get("AnswerIntent", "")).strip()
            qcode = str(row.get("QuestionCode", "")).strip()[:150]
            qcomment = str(row.get("QuestionComment", "")).strip()[:150]
            acode = str(row.get("AnswerCode", "")).strip()[:150]
            acomment = str(row.get("AnswerComment", "")).strip()[:150]

            context_parts.append(f"""
Similarity: {sim:.2f}
QuestionIntent: {qintent}
AnswerIntent: {aintent}

QuestionCode: {qcode}
QuestionComment: {qcomment}
AnswerCode: {acode}
AnswerComment: {acomment}
""")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
Generate a one-line meaningful comment for the following code (intent: {input_intent}):

{input_code}

Use the following similar examples as context:

{context}
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # switched from GPT-5 Mini
            messages=[
                {"role": "system", "content": "You are an expert programmer who writes concise, meaningful code comments."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=150  # note: use 'max_completion_tokens' for GPT-4o-mini
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during generation: {e}")
        return ""


#  METRIC SETUP

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1


#  MAIN LOOP

for idx, row in tqdm(goal_df.iterrows(), total=len(goal_df)):
    code = str(row["code"])
    intent = str(row["label"])
    reference_comment = str(row["comment"]).strip()

    generated_comment = generate_comment(code, intent)
    if not generated_comment:
        generated_comment = "No comment generated."

    goal_df.at[idx, "GeneratedComment"] = generated_comment

    ref_tokens = reference_comment.split()
    cand_tokens = generated_comment.split()

    # BLEU   
    bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)
    # ROUGE-L
    rouge_l = rouge.score(reference_comment, generated_comment)["rougeL"].fmeasure
    # METEOR
    meteor = meteor_score([ref_tokens], cand_tokens)

    goal_df.at[idx, "BLEU"] = bleu
    goal_df.at[idx, "ROUGE_L"] = rouge_l
    goal_df.at[idx, "METEOR"] = meteor

    # Save after each row
    goal_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n Saved progress to {OUTPUT_FILE}")
print(" Done generating comments and metrics!")
