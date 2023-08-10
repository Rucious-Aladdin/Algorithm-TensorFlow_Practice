from corpus import preprocess
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
import numpy as np


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
                
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x, y, eps = 1e-8):
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print("%s를 찾을 수 없습니다." % query)
        return
    
    print("\n[query]" + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
        
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        
        print(" %s: %s" % (id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            return

## PMI 정의  -> p(x, y){joint probability} / p(x) * p(y) = PMI(x, y)
# PPMI -> PPMI(x, y) = max(0, PMI(x, y))

def ppmi(C, verbose=True, eps=1e-8):
    """_summary_

    Args:
        C (_type_): co_occurence matrix(동시발생 행렬, numpy matrix)
        verbose (bool, optional): 진행 상황 출력 여부
        eps (_type_, optional): log내의 수가 0이 되는 것 방지. Defaults to 1e-8.
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)
            
            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print("%.1f%% 완료" % (100 * cnt/total))
    return M


if __name__ == "__main__":
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    C = create_co_matrix(corpus, len(word_to_id))
    c0 = C[word_to_id["you"]]
    c1 = C[word_to_id["i"]]
    print(cos_similarity(c0, c1))

    most_similar("you", word_to_id, id_to_word, C, top=5)


    C = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.int32)

    W = ppmi(C)
    print(C)
    print(W)
        