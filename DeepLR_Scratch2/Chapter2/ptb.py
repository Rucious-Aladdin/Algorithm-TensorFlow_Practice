from corpus import preprocess
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
import numpy as np
import matplotlib.pyplot as plt
from corpus import preprocess
from co_matrix import cos_similarity, most_similar, create_co_matrix, ppmi
from dataset1 import ptb

window_size = 2
word_vec_size = 100
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

C = create_co_matrix(corpus, vocab_size, window_size)
W = ppmi(C, verbose=True)

try:
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=word_vec_size, n_iter=5, random_state=None)
except:
    U,S,V = np.linalg.svd(W)

word_vecs = U[:, :word_vec_size]
querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)