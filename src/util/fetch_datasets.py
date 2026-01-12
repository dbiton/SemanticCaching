import os

# Increase timeout to 100 seconds (default is 10)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Disable fast transfer if enabled, sometimes causes issues
os.environ["HF_HUB_ETAG_TIMEOUT"] = "100"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"

import pickle
import numpy as np
from typing import *
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from datasets.combine import concatenate_datasets
from scipy.cluster.hierarchy import DisjointSet
from sklearn.preprocessing import normalize

embeds_dir = "datasets"

models: Dict[str, SentenceTransformer] = {}

def embed_strings(strings: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if model_name not in models:
        models[model_name] = SentenceTransformer(model_name)
        print(f"Loaded model {model_name} into device {models[model_name].device}")
    model = models[model_name]
    embs = model.encode(strings, convert_to_numpy=True, show_progress_bar=True)
    return embs

@contextmanager
def writer(path: str):
    if os.path.exists(path):
        f = None
    else:
        f = NamedTemporaryFile(mode="wb", delete=False)
        tmp_path = f.name
    try:
        yield f
        if f is not None:
            f.close()  # Ensure closed before rename
            os.rename(tmp_path, path)
    except:
        if f is None:
            print(f'Skipping "{path}" because it already exists')
        else:
            f.close()
            os.remove(tmp_path)
            raise

def pack_and_dump(path: str, texts: List[str], meta: Dict[str, List[Any]], model_name: str = "all-MiniLM-L6-v2"):
    assert all(len(v) == len(texts) for v in meta.values()), "Metadata length mismatch"
    with writer(path) as f:
        if f is None:
            print(f'Skipping "{path}" because it already exists')
            return
        embeds = embed_strings(texts, model_name=model_name)
        normalized_embeds = normalize(embeds)
        payload = {
            "text": texts,
            "normalized_embeds": normalized_embeds,
            "embeds": embeds,
            "meta": meta
        }
        pickle.dump(payload, f)

def build_eli5() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("sentence-transformers/eli5", trust_remote_code=True)
    texts: List[str] = ds["train"]["question"]
    meta = {
    }
    return texts, meta

def build_wildchat() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("allenai/WildChat-1M")
    texts, session_id, turn_id = [], [], []
    sid = 0
    for ex in ds["train"]:
        if ex.get("language") != "English":
            continue
        # Keep only user turns; preserve order
        user_turns = []
        for t in ex.get("conversation", []):
            role = t.get("role") or t.get("author_role")  # robustness
            if role in ("user", "User"):
                content = t.get("content") or t.get("text")
                if content:
                    user_turns.append(content)
        if not user_turns:
            continue
        for k, ut in enumerate(user_turns):
            texts.append(ut)
            session_id.append(f"wildchat_{sid}")
            turn_id.append(k)
        sid += 1
    meta = {
        "session_id": session_id,
        "turn_id": turn_id,
    }
    return texts, meta

def build_qrecc() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("svakulenk0/qrecc")
    texts, session_id, turn_id = [], [], []
    # Fields: 'conversation_id' (or 'qid_conv'), 'turn_id' (or 'turn'), 'question'
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            q = ex.get("question") or ex.get("rewritten_question") or ex.get("raw_question")
            if not q:
                continue
            cid = ex.get("conversation_id") or ex.get("qid_conv") or ex.get("conversation_no") or "unk"
            tid = ex.get("turn_id") or ex.get("turn") or 0
            texts.append(q)
            session_id.append(f"qrecc_{cid}")
            turn_id.append(int(tid))
    # Sort by (session, turn) to preserve order
    order = sorted(range(len(texts)), key=lambda i: (session_id[i], turn_id[i]))
    texts = [texts[i] for i in order]
    session_id = [session_id[i] for i in order]
    turn_id = [turn_id[i] for i in order]
    meta = {
        "dataset": ["QReCC"] * len(texts),
        "session_id": session_id,
        "turn_id": turn_id,
        "language": ["en"] * len(texts),
        "timestamp": [None] * len(texts),
        "pair_id": [None] * len(texts),
    }
    return texts, meta

def build_stackoverflow() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("pacovaldez/stackoverflow-questions")
    # Can also get the questions themselves
    texts = concatenate_datasets([ds[s] for s in ('train','validation','test')])['title']
    return texts, meta

def build_quora() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("quora", trust_remote_code=True)
    # structure: each row has 'questions': [{'text': q1}, {'text': q2}], possibly 'id'
    texts, question_id = [], []
    union_find = DisjointSet()
    for ex in ds["train"]:
        q1, q2 = ex['questions']['text']
        q1_id, q2_id = ex['questions']['id']
        union_find.add(q1_id)
        union_find.add(q2_id)
        if ex['is_duplicate']:
            union_find.merge(q1_id, q2_id)
        texts += [q1, q2]
        question_id += [q1_id, q2_id]
    set_id = [union_find[qid] for qid in question_id]
    meta = {
        "set_id": set_id,
        "question_id": question_id
    }
    return texts, meta

def build_mmlu() -> Tuple[List[str], Dict[str, List[Any]]]:
    # MMLU (Massive Multitask Language Understanding)
    # The 'all' config loads all subjects (STEM, humanities, etc.)
    print("Loading MMLU (config: 'all')...")
    try:
        ds = load_dataset("cais/mmlu", "all")
    except:
        # Fallback if 'all' config isn't immediately found in some versions
        ds = load_dataset("cais/mmlu", "auxiliary_train")
        
    texts = []
    # Iterate over available splits (usually 'test', 'validation', 'dev', 'auxiliary_train')
    for split in ds:
        if "question" in ds[split].features:
            texts.extend(ds[split]["question"])
            
    meta = {}
    return texts, meta

def build_hotpotqa() -> Tuple[List[str], Dict[str, List[Any]]]:
    # HotpotQA (Multi-hop reasoning)
    print("Loading HotpotQA...")
    ds = load_dataset("hotpot_qa", "distractor")    
    texts = []
    # Splits: 'train', 'validation'
    for split in ds:
        texts.extend(ds[split]["question"])
        
    meta = {}
    return texts, meta

def build_triviaqa() -> Tuple[List[str], Dict[str, List[Any]]]:
    # TriviaQA
    # 'rc' (Reading Comprehension) is the standard configuration
    print("Loading TriviaQA (config: 'rc')...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")
    
    texts = []
    # Splits: 'train', 'validation', 'test'
    for split in ds:
        texts.extend(ds[split]["question"])
        
    meta = {}
    return texts, meta

def build_nq() -> Tuple[List[str], Dict[str, List[Any]]]:
    texts = []
    ds = load_dataset("nq_open")
    for split in ("train", "validation"):
        texts.extend([q for q in ds[split]["question"] if q])
    meta = {
    }
    return texts, meta

def build_msmarco() -> Tuple[List[str], Dict[str, List[Any]]]:
    texts = []
    ds = load_dataset("ms_marco", "v2.1")
    for split in ("train", "validation", "test"):
        texts.extend([q for q in ds[split]["query"] if q])
    meta = {
    }
    return texts, meta

if __name__ == "__main__":
    os.makedirs(embeds_dir, exist_ok=True)

    print("generating TriviaQA...")
    texts, meta = build_triviaqa()
    pack_and_dump(os.path.join(embeds_dir, "embeds_triviaqa.pkl"), texts, meta)

    print("generating HotpotQA...")
    texts, meta = build_hotpotqa()
    pack_and_dump(os.path.join(embeds_dir, "embeds_hotpotqa.pkl"), texts, meta)
    
    print("generating MMLU...")
    texts, meta = build_mmlu()
    pack_and_dump(os.path.join(embeds_dir, "embeds_mmlu.pkl"), texts, meta)
    
    print("generating ELI5...")
    texts, meta = build_eli5()
    pack_and_dump(os.path.join(embeds_dir, "embeds_eli5.pkl"), texts, meta)

    print("generating WildChat...")
    texts, meta = build_wildchat()
    pack_and_dump(os.path.join(embeds_dir, "embeds_wildchat.pkl"), texts, meta)

    print("generating Natural Questions...")
    texts, meta = build_nq()
    pack_and_dump(os.path.join(embeds_dir, "embeds_nq.pkl"), texts, meta)

    print("generating MS MARCO...")
    texts, meta = build_msmarco()
    pack_and_dump(os.path.join(embeds_dir, "embeds_msmarco.pkl"), texts, meta)

    print("generating StackOverflow...")
    texts, meta = build_stackoverflow() if 'meta' in locals() else build_stackoverflow()[0], {} 
    pack_and_dump(os.path.join(embeds_dir, "embeds_stackoverflow.pkl"), texts, meta)

    print("generating Quora Question Pairs...")
    texts, meta = build_quora()
    pack_and_dump(os.path.join(embeds_dir, "embeds_quora_qp.pkl"), texts, meta)

    print("Done.")