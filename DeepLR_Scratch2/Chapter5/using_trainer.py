import numpy as np
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
from common.optimizer import SGD
from dataset1 import ptb
from RNNLM import SimpleRNNLM
import matplotlib.pyplot as plt
from common.trainer import RnnlmTrainer

if __name__ == "__main__":

    batch_size = 10
    wordvec_size = 100
    hidden_size = 100
    time_size = 5
    lr = 0.1
    max_epoch = 150

    corpus, word_to_id, id_to_word = ptb.load_data("train")
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1) 

    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    print("말뭉치 크기: %d, 어휘수: %d" % (corpus_size, vocab_size))

    max_iters = data_size // (batch_size * time_size)
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []


    model = SimpleRNNLM(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)
    
    trainer.fit(xs, ts, max_epoch, batch_size, time_size)
    trainer.plot()