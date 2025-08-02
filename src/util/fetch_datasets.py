from itertools import chain
import os
import pickle
import numpy as np
from typing import *
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from datasets.combine import concatenate_datasets
from scipy.cluster.hierarchy import DisjointSet

embeds_dir = "datasets"

models: Dict[str, SentenceTransformer] = {}

def embed_strings(strings: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if model_name not in models:
        models[model_name] = SentenceTransformer(model_name)
        print(f"Loaded model {model_name} into device {models[model_name].device}")
    model = models[model_name]
    embs = model.encode(strings, convert_to_numpy=True, show_progress_bar=True)
    return embs.astype(np.float16)  # space saver

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
        payload = {
            "text": texts,
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

    print("generating ELI5...") #V
    texts, meta = build_eli5()
    pack_and_dump(os.path.join(embeds_dir, "embeds_eli5.pkl"), texts, meta)

    print("generating WildChat...") #V
    texts, meta = build_wildchat()
    pack_and_dump(os.path.join(embeds_dir, "embeds_wildchat.pkl"), texts, meta)

    print("generating Natural Questions...") #V
    texts, meta = build_nq()
    pack_and_dump(os.path.join(embeds_dir, "embeds_nq.pkl"), texts, meta)

    print("generating MS MARCO...") #V
    texts, meta = build_msmarco()
    pack_and_dump(os.path.join(embeds_dir, "embeds_msmarco.pkl"), texts, meta)

    print("generating StackOverflow...") #V
    texts, meta = build_stackoverflow()
    pack_and_dump(os.path.join(embeds_dir, "embeds_stackoverflow.pkl"), texts, meta)

    print("generating Quora Question Pairs...") #V
    texts, meta = build_quora()
    pack_and_dump(os.path.join(embeds_dir, "embeds_quora_qp.pkl"), texts, meta)

    print("Done.")
