from typing import Iterator, Tuple, List, Dict


sentence = List[str]


def word2features(sent: sentence, i: int) -> Dict:
	# todo modificar los features para obtener mayor presicion con un contexto
	#  mas grande
	word = sent[i]

	features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
	if i > 0:
		word1 = sent[i - 1]
		features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
		})
	else:
		features['BOS'] = True

	if i < len(sent) - 1:
		word1 = sent[i + 1]
		features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
	else:
		features['EOS'] = True

	return features


def sent2features(sent: sentence):
	return [word2features(sent, i) for i in range(len(sent))]



if __name__ == '__main__':
	from itertools import chain
	import nltk
	import sklearn
	import scipy.stats
	from sklearn.metrics import make_scorer
	from sklearn.model_selection import cross_val_score
	from sklearn.model_selection import RandomizedSearchCV
	from sklearn_crfsuite import scorers
	import sklearn_crfsuite
	from sklearn_crfsuite import metrics
	import utils

	# todo seria bueno hacer algo como randomizedsearch para buscar mejor los
	#  params del estimador, error de params

	words, tags = utils.transform_tsv2BIO('corpus.tsv')
	tags = utils.clean_tags(tags)

	sent_get = utils.SentenceGetter(words=words)
	w_train, w_test = sent_get.split(-20)
	t_train, t_test = tags[:len(w_train)], tags[len(w_train):]

	w_train_f = sent2features(w_train)
	w_test_f = sent2features(w_test)

	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs',
		c1=0.1,
		c2=0.1,
		max_iterations=100,
		all_possible_transitions=True
	)
	# crf.fit recibe una lista de listas de feature vectors, y una listas de
	# listas de tags, cada lista se toma como un documento
	crf.fit([w_train_f], [t_train])

	# crf.predict recibe una lista de listas de features vectors y devuelve una
	# lista de listas de tags, similar al fit
	t_pred: List[List[str]] = crf.predict(w_test_f)

	print(metrics.flat_f1_score(t_test, t_pred, average='weighted'))

	print()
	params_space = {'c1': scipy.stats.expon(scale=0.5),
					'c2': scipy.stats.expon(scale=0.05),
	}

	f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')

	rs = RandomizedSearchCV(crf, params_space, cv=3, verbose=1, n_jobs=4,
							n_iter=50,
							scoring=f1_scorer)
	rs.fit(w_train_f, t_train)

	# print('best params:', rs.best_params_)
	# print('best CV score:', rs.best_score_)
	# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
	#
	# crf = rs.best_estimator_
	# t_pred = crf.predict([w_test_f])
	# print(metrics.flat_classification_report(
	# 	t_test, t_pred, digits=3))

	t_pred2 = crf.predict([sent2features(['el', 'Ministerio', 'del', 'Interior',
										  'y', 'la', 'Unión', 'de', 'Jovenes',
										  'Capitalistas', 'de', 'Chile', 'participan',
										  'en', 'la', 'recogida', 'de',
										  'materias', 'primas', '.'])])
	print(t_pred2)
	# todo predecir localizaciones tambien
	# todo instalar word2vec para tirar los embeddings en crf y en redes