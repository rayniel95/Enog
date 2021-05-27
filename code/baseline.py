import joblib
import sklearn.base
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import learning_curve

import utils


class MemoryTagger(sklearn.base.BaseEstimator):

	def fit(self, X, y):
		'''
		Expects a list of words as X and a list of tags as y.
		'''
		voc = {}
		self.tags = []
		for x, t in zip(X, y):
			if t not in self.tags:
				self.tags.append(t)
			if x in voc:
				if t in voc[x]:
					voc[x][t] += 1
				else:
					voc[x][t] = 1
			else:
				voc[x] = {t: 1}
		self.memory = {}
		for k, d in voc.items():
			self.memory[k] = max(d, key=d.get)

	def predict(self, X, y=None):
		'''
		Predict the the tag from memory. If word is unknown, predict 'O'.
		'''
		return [self.memory.get(x, 'O') for x in X]



# words, tags = utils.transform_tsv2BIO('corpus.tsv')
words, tags = joblib.load('Corpus2')

all_words = list(set(words))
all_tags = list(set(tags))

n_words = len(all_words)
n_tags = len(all_tags)


sent_get = utils.SentenceGetter(words=words)
w_train, w_test = words[:-1000], words[-1000:]
t_train, t_test = tags[:len(w_train)], tags[len(w_train):]

# f1: 99
# precision: 99
model = MemoryTagger()
model.fit(w_train, t_train)

pred = model.predict(w_test)

print(precision_score(t_test, pred, average='weighted'))
