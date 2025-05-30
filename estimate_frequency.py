
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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # small & fast model


nltk.download('brown')
nltk.download('punkt')
nltk.download('punkt_tab')


corpus_words = brown.words()
word_freq = Counter([w.lower() for w in corpus_words])
total_words = sum(word_freq.values())
word_prob = {word: freq / total_words for word, freq in word_freq.items()}

model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

def calculate_surprisal(sentence: str):
    tokens = nltk.word_tokenize(sentence.lower())
    score = -sum(math.log(word_prob.get(word, 1e-7)) for word in tokens)
    return score

def generate_dataset(output_path: str) -> None:
    ds_chat = load_dataset("allenai/WildChat-1M")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i_e, e in enumerate(ds_chat['train']):
            print(i_e)
            try:
                if e['language'] == 'English':
                    conv = e['conversation']
                    for i in range(0, len(conv), 2):
                        prompt = conv[i]
                        sentence = prompt['content']
                        is_toxic = prompt['toxic']
                        is_redacted = prompt['redacted']
                        country = prompt['country']
                        state = prompt['state']
                        role = prompt['role']
                        # perplexity = calculate_perplexity(sentence)
                        surprisal = calculate_surprisal(sentence)
                        # embedding = calculate_embedding(sentence)
                        entry = {
                            #"sentence": sentence,
                            #"embedding": embedding,
                            #"role": role,
                            #"conv_index": i / 2,
                            #"perplexity": perplexity,
                            "surprisal": surprisal,
                            "num_words": len(sentence.split()),
                            "num_chars": len(sentence),
                            #"is_toxic": is_toxic,
                            #"is_redacted": is_redacted,
                            #"country": country,
                            #"state": state
                        }
                        json.dump(entry, f_out)
                        f_out.write('\n')
            except Exception as e:
                print(e)

def create_embeds_covers(embeds, same_embed_distance):
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
    return [int(v) for v in similar_counts]

def generate_results(input_path, output_path):
    SAME_EMBED_DISTANCE = 0.5
    embeddings = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i_line, line in enumerate(f):
            entry = ujson.loads(line)
            embedding = entry['embedding']
            embeddings.append(embedding)
    covers = create_embeds_covers(embeddings, SAME_EMBED_DISTANCE)
    with open(output_path, 'w') as f_out:
        ujson.dump(covers, f_out)

import matplotlib.pyplot as plt


def train(ds_path, res_path):
    X = []
    y = []
    with open(ds_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = ujson.loads(line)
            X.append(entry)
    with open(res_path, 'r', encoding='utf-8') as f:
        y = ujson.load(f)
    y = [v/len(y) for v in y]
    X = pd.DataFrame(X)
    X = X[['uniqueness']] # 0.6801
    # X = X[['uniqueness', 'num_words', 'num_chars']] 0.8493 (perplexity seems to be the most useful)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    
    
    

# generate_dataset("ds.json")
# generate_results("ds.json", "res.json")
train("ds.json", "res.json")
