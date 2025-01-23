from datasets import load_dataset

# In order of max Hit Rate: so, chat, bing 

ds_so = load_dataset("pacovaldez/stackoverflow-questions")
ds_chat = load_dataset("allenai/WildChat-1M")
ds_bing = load_dataset("corbyrosset/researchy_questions")
