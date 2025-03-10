import os
import pickle
import random
import pandas as pd
import tqdm
from typing import *
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

embeds_dir = "datasets"

# dataset specific params
steam_limit = 100000
steam_seed = 0

models: Dict[str, SentenceTransformer] = dict()
def embed_strings(strings: List[str], model_name='all-MiniLM-L6-v2'):
    if model_name not in models:
        models[model_name] = SentenceTransformer(model_name)
        print(f"Loaded model {model_name} into device {models[model_name].device}")
    model = models[model_name]
    embeddings = model.encode(strings, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def skip_all_except(selected_rows: Iterable[int], *, show_progress: bool = True):
    selected_rows = set(selected_rows)
    sample_size = len(selected_rows)
    if show_progress:
        progress = tqdm.tqdm(total=sample_size)
        def f(i):
            if i in selected_rows:
                progress.update(1)
                return False
            else:
                return True
    else:
        def f(i):
            return i not in selected_rows
    return f

def sample_from_csv(file_path: str, total_rows: int, sample_size: int, *, show_progress: bool = True, seed: int = None, header: Optional[int] = 0, names: Optional[Sequence[str]] = None, dtype: Optional[Dict] = None) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)
    selected_rows = set(random.sample(range(1, total_rows), sample_size))
    if isinstance(header, int):
        assert header < total_rows, f"Header is {header} but total_rows is {total_rows}"
        selected_rows.add(header)
    else:
        assert names is not None, f"Names must be provided if there is no header"
    df = pd.read_csv(
        file_path,
        header=header,
        names=names,
        skiprows=skip_all_except(selected_rows, show_progress=show_progress),
        nrows=len(selected_rows),
        encoding_errors="ignore",
        dtype=dtype,
    )
    return df

@contextmanager
def writer(path):
    if os.path.exists(path):
        f = None
    else:
        f = NamedTemporaryFile(mode="wb", delete=False)
        tmp_path = f.name
    try:
        yield f
        f.close()  # Ensure the file is closed before renaming
        os.rename(tmp_path, path)
    except:
        if f is None:
            print(f"Skipping \"{path}\" because it already exists")
        else:
            f.close()
            os.remove(tmp_path)
            raise

if __name__ == "__main__":
    print("generating bing...")
    with writer(os.path.join(embeds_dir, "embeds_bing.pkl")) as f:
        if f is not None:
            ds_bing = load_dataset("corbyrosset/researchy_questions")
            questions_bing = ds_bing['train']['question']
            embeds_bing = embed_strings(questions_bing)
            pickle.dump(embeds_bing, f)

    print("generating so...")
    with writer(os.path.join(embeds_dir, "embeds_so.pkl")) as f:
        if f is not None:
            ds_so = load_dataset("pacovaldez/stackoverflow-questions")
            questions_so = ds_so['train']['title']
            embeds_so = embed_strings(questions_so)
            pickle.dump(embeds_so, f)

    print("generating chat...")
    with writer(os.path.join(embeds_dir, "embeds_chat.pkl")) as f:
        if f is not None:
            ds_chat = load_dataset("allenai/WildChat-1M")
            questions_chat = [e['conversation'][0]['content'] for e in ds_chat['train'] if e['language'] == "English"]
            embeds_chat = embed_strings(questions_chat)
            pickle.dump(embeds_chat, f)
    
    print("generating steam...")
    with writer(os.path.join(embeds_dir, "embeds_steam.pkl")) as f:
        if f is not None:
            import kagglehub
            print(f"Downloading dataset if needed... This can take a long time if the raw dataset has not been downloaded!")
            path = kagglehub.dataset_download("kieranpoc/steam-reviews/versions/2")
            print(f"Dataset is available at {path}")
            csv_path = os.path.join(path, "all_reviews", "all_reviews.csv")
            print(f"Sampling from full: {csv_path}")
            df = sample_from_csv(csv_path, 113883709, steam_limit, seed=steam_seed)
            df.sort_values("timestamp_created", inplace=True)
            df.reset_index(drop=True, inplace=True)
            reviews_steam = df['review']
            embeds_steam = embed_strings(reviews_steam)
            pickle.dump(embeds_steam, f)
