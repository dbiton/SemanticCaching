import math
from wordfreq import word_frequency, tokenize

def calculate_surprisal(text: str):
    wordlist = "large"
    lang = 'en'
    floor = 1e-9 # large wordlist uses 100M words, so min frequency is 1/100M = 1e-8
    tokens = [t.lower() for t in tokenize(text, lang, include_punctuation=False)]
    freqs = [word_frequency(t, lang, minimum=floor, wordlist=wordlist) for t in tokens]
    score = -sum(math.log2(f) for f in freqs)
    return score