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
	import scipy.stats
	from sklearn.metrics import make_scorer, f1_score
	from sklearn.model_selection import RandomizedSearchCV, learning_curve
	import sklearn_crfsuite
	import utils
	from sklearn.externals import joblib
	import numpy


	features = ['bias', 'word.lower()', 'word[-3:]', 'word[-2:]',
				'word.isupper()', 'word.istitle()', 'word.isdigit()',
				'-1:word.lower()', '-1:word.istitle()', '-1:word.isupper()',
				'BOS', '+1:word.lower()', '+1:word.istitle()',
				'+1:word.isupper()', 'EOS']

	words, tags = utils.transform_tsv2BIO('corpus.tsv')

	sent_get = utils.SentenceGetter(words=words)
	w_train, w_test = sent_get.split(-20)
	t_train, t_test = tags[:len(w_train)], tags[len(w_train):]

	w_train_f = sent2features(w_train)
	w_test_f = sent2features(w_test)

	# crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
	# 						   max_iterations=100, all_possible_transitions=True
	# )
	#
	# print()
	# params_space = {'c1': scipy.stats.expon(scale=0.5),
	# 				'c2': scipy.stats.expon(scale=0.05),
	# }
	#
	# f1_scorer = make_scorer(f1_score, average='weighted')
	# todo ver el warning que da que es resultado del f1 score
	# rs = RandomizedSearchCV(crf, params_space, verbose=1, cv=3,
	# 						n_jobs=4, n_iter=200, scoring=f1_scorer)
	# rs.fit(w_train_f, t_train)
	#
	# print('best params:', rs.best_params_)
	# print('best CV score:', rs.best_score_)
	# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
	#
	# crf = rs.best_estimator_
	# joblib.dump(crf, 'crf_model')
	crf = joblib.load('crf_model')
	# t_pred2 = crf.predict(sent2features(['el', 'Ministerio', 'del', 'Interior',
	# 									  'y', 'la', 'Union', 'de', 'Jovenes',
	# 									  'Capitalistas', 'de', 'Chile',
	# 									  'participan', 'en', 'la', 'recogida',
	# 									  'de', 'materias', 'primas', '.']))
	# print(t_pred2)
	# new = utils.to_feature_array(features, w_train_f)
	# learning_curve(crf, new, t_train, cv=3)
	pred = crf.predict(w_test_f)

	print(f1_score(t_test, pred, average='weighted'))
	# todo instalar word2vec para tirar los embeddings en crf y en redes
	# todo falta evaluar el modelo, ver una learning curve
	# todo hacer un clasification report con el test set
	# todo entrenar con el corpus de ronald