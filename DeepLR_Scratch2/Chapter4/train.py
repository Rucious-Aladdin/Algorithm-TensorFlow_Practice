import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2\\Chapter4")
from common import config
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from CBOW_withNAG import CBOW
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset1 import ptb
import numpy as np


window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 30
eval_interval = 50

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
print(corpus.shape)
print(len(word_to_id))

contexts, target = create_contexts_target(corpus, window_size)
"""
contexts_train = contexts[:-200000]
target_train = target[:-200000]

contexts_val = contexts[-200000:-100000]
target_val = target[-200000:-100000]
"""
contexts_train = contexts[:5000]
target_train = target[:5000]

contexts_val = contexts[-1000:]
target_val = target[-1000:]


print("contexts_train.shape: "  + str(contexts_train.shape))
print("target_train.shape: "  + str(target_train.shape))

print("contexts_val.shape: "  + str(contexts_val.shape))
print("target_val.shape: "  + str(target_val.shape))


if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer, patience_check=1)


trainer.fit(contexts_train, target_train, contexts_val, target_val, max_epoch, batch_size, eval_interval=eval_interval)
trainer.plot()

word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)

params = {}
params["word_vecs"] = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word
pkl_file = "cbow_params.pkl"

with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)