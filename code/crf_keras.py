import joblib
from keras.models import Sequential
from keras.layers import Embedding, Input
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score



def make_model():
	model = Sequential()
	model.add(Embedding(input_dim=n_words + 1, output_dim=20,
						input_length=max_len))

	crf = CRF(n_tags)
	model.add(crf)

	model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
	return model

# words, tags = utils.transform_tsv2BIO('corpus.tsv')
words, tags = joblib.load('Corpus2')

all_words = list(set(words))
all_tags = list(set(tags))

n_words = len(all_words)
n_tags = len(all_tags)

sent_get = utils.SentenceGetter(words=words)
w_train, w_test = sent_get.split(-20)
t_train, t_test = tags[:len(w_train)], tags[len(w_train):]

#PREPARING THE DATA///////////
max_len = 75
word2dix = {w: i + 1 for i, w in enumerate(all_words)}
tag2dix = {t: i for i, t in enumerate(all_tags)}

train_sent_w = sent_get.to_sentences(w_train)
train_sent_t = [[t_train.pop(0) for _ in sent] for sent in train_sent_w]

x = [[word2dix[w] for w in s] for s in train_sent_w]
x = pad_sequences(truncating='post', maxlen=max_len, sequences=x,
				  padding='post', value=n_words-1)

y = [[tag2dix[t] for t in s] for s in train_sent_t]
y = pad_sequences(maxlen=max_len, sequences=y, truncating='post',
				  padding='post', value=tag2dix['O'])

y = [to_categorical(i, num_classes=n_tags) for i in y]

x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)


model = make_model()

history = model.fit(x_tr, np.array(y_tr), batch_size=32, epochs=20,
					validation_split=0.1, verbose=1)
pred = model.predict(x_te)

print('pppppppppppppppp')
print(utils.cross_val(make_model, x_tr, y_tr))
pred2 = utils.to_tags(pred, tag2dix)
y_te2 = utils.to_tags(y_te, tag2dix)

print(precision_score(y_te2, pred2, average='weighted'))
print(f1_score(y_te2, pred2, average='weighted'))
# f1: 94
# precision: 98
hist = pd.DataFrame(history.history)

print('000000000000000000')
#Learning Curve////////////
print(hist)
plt.style.use("ggplot")
plt.figure(figsize=(30, 30,), )

plt.plot(hist["crf_viterbi_accuracy"])
plt.plot(hist["val_crf_viterbi_accuracy"])
# plt.plot(hist)
plt.show()