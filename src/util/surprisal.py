
'''
import json
from datasets import load_dataset
import faiss
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math
import nltk
from nltk.corpus import brown
from collections import Counter
import math
from sentence_transformers import SentenceTransformer
import ujson
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from typing import List

nltk.download('brown')
nltk.download('punkt')
nltk.download('punkt_tab')

corpus_words = brown.words()
word_freq = {}#Counter([w.lower() for w in corpus_words])
total_words = sum(word_freq.values())
word_prob = {word: freq / total_words for word, freq in word_freq.items()}

model_name = "distilgpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).eval().to(device)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # small & fast model

def calculate_perplexity(sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)  # Move input to GPU
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = math.exp(loss.item())
    return perplexity

def calculate_embedding(sentence: str):
    return embedding_model.encode(sentence, convert_to_numpy=True).tolist()
    
    
def generate_dataset(limit: int) -> List[str]:
    dataset = load_dataset("PaulPauls/openwebtext-sentences", split="train")
    return dataset[:limit]['text']

def calc_props(ds: List[str]) -> None:
    result = []
    for s in ds:
        surprisal = calculate_surprisal(s)
        entry = {
            "surprisal": surprisal,
            "num_words": len(s.split()),
            "num_chars": len(s),
        }
        result.append(entry)
    return pd.DataFrame(result)

def create_X_y():
    ds = generate_dataset(1000000)
    embeds = embedding_model.encode(ds, convert_to_numpy=True, show_progress_bar=True)
    counts = calc_embeds_counts(embeds, .5) / len(embeds)
    props = calc_props(ds)
    return props, counts

def calc_embeds_counts(embeds, same_embed_distance):
    embeds = np.asarray(embeds, dtype=np.float32)
    n, d = embeds.shape
    cpu_index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    else:
        index = cpu_index
    index.add(embeds)
    threshold = same_embed_distance ** 2
    lims, _, indices = index.range_search(embeds, threshold)
    similar_counts = np.array([
        np.sum(indices[lims[i]:lims[i+1]] != i)  # exclude self-match
        for i in range(n)
    ])
    return np.array([int(v) for v in similar_counts])

def train():
    X, y = create_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    reg = lgb.LGBMRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"R^2 score: {score:.4f}")
    importances = reg.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    feat_df.head(20).plot(kind='barh', x='feature', y='importance', figsize=(8, 6), legend=False)
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
'''

def calculate_perplexity(sentence: str):
    raise Exception("don't use!")

def calculate_surprisal(sentence: str):
    raise Exception("don't use!")
    tokens = nltk.word_tokenize(sentence.lower())
    score = -sum(math.log(word_prob.get(word, 1e-7)) for word in tokens)
    return score