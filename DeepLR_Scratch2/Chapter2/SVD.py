from corpus import preprocess
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
import numpy as np
import matplotlib.pyplot as plt
from corpus import preprocess
from co_matrix import cos_similarity, most_similar, create_co_matrix, ppmi

if __name__ == "__main__":
    text = "You say goodbye and I say Hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(id_to_word)
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    W = ppmi(C)

    U, S, V = np.linalg.svd(W)

    print(U, S, V)
    print(U.shape)
    print(S.shape)
    print(V.shape)
    print(np.identity(7) * S)

    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
        
    plt.scatter(U[:,0], U[:,1], alpha=0.5)
    plt.show()