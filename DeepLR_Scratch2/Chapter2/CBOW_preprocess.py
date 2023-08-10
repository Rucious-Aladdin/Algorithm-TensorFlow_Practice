import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
import numpy as np
from common.layers import MatMul
from corpus import preprocess
from common.util import convert_one_hot

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0: # t==0이면 target단어 이므로 제외함.
                continue
            cs.append(corpus[idx + t]) #for 문을 통해 앞뒤 단어 추가.
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)

if __name__ == "__main__":
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)

    print(id_to_word)
    contexts, target = create_contexts_target(corpus, window_size=1)
    print(contexts)
    print(target)
    vocab_size = len(word_to_id)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)
    print(target.shape)
    print(contexts.shape)
    