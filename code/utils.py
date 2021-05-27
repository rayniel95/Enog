from typing import List, Tuple, Dict, Any

import keras
import sklearn_crfsuite
import numpy as np



def read_tsv(name: str):
	data = None
	with open(name, 'r', encoding='utf-8') as file:
		data = file.read()
	return data


def transform_tsv2BIO(name: str) -> Tuple[List[str], List[str]]:
	words = []
	tags = []
	actual: str = ''
	with open(name, 'r', encoding='utf-8') as file:
		for item in file.readlines():
			word: str
			tag: str
			word, tag = item.replace('\n', '').split('\t')
			if tag != 'O':
				words_in = word.split()
				words.append(words_in.pop(0))
				tags.append(f'B-{tag}')
				for word_in in words_in:
					words.append(word_in)
					tags.append(f'I-{tag}')
			else:
				words.append(word)
				tags.append(tag)

	return words, tags


def del_not_org_tags(tags: List[str]) -> List[str]:
	n_tags = []
	tag: str
	for tag in tags:
		if tag != 'O' and tag.split('-')[1] != 'ORG':
			n_tags.append('O')
		else: n_tags.append(tag)
	return n_tags


def to_categories(a_list: List[str]) -> List[int]:
	cats = {}
	index = 0
	other_list = []
	for tag in a_list:
		try:
			other_list.append(cats[tag])
		except:
			cats[tag] = index
			index += 1
			other_list.append(cats[tag])

	return other_list


def to_feature_array(feature_names: List[str], feature_vectors: List[Dict[str, Any]]):

	new_features = []
	for vector in feature_vectors:
		new_features.append([vector.get(name, 0) for name in feature_names])

	return new_features



class SentenceGetter:

	def __init__(self, **kwargs):
		if kwargs.get('words', False):
			self.text = self.to_sentences(kwargs['words'])
		else: raise Exception('incorrect arguments')

	@staticmethod
	def to_sentences(sents: List[str]) -> List[List[str]]:
		resp = []
		sents_cop = sents
		for times in range(sents.count('.')):
			part = sents_cop[: sents_cop.index('.') + 1]
			resp.append(part)
			sents_cop = sents_cop[sents_cop.index('.') + 1:]
		return resp

	def split(self, number: int) -> Tuple[List[str], List[str]]:
		part1, part2 = self.text[: number], self.text[number:]
		t1 = []
		ret1 = [t1.extend(ls) for ls in part1][0]
		t2 = []
		ret2 = [t2.extend(ls) for ls in part2][0]

		return t1, t2


def to_tags(vectors, tags_index: dict):
	vektor = list(vectors)
	tags = []
	index2tags = {value: key for key, value in tags_index.items()}
	for sentence in vektor:
		sent = []
		for vec in sentence:
			sent.append(index2tags[list(vec).index(1.0)])
		tags.append(sent)

	new_tags = []

	for el in tags:
		new_tags.extend(el)

	return new_tags


def cross_val(model_maker, train: np.ndarray, target: list, cv=3, epochs=30,
			  batch_size=32, validation_split=0.1):
	model: keras.Model
	num_validation_samples = len(train) // cv
	validation_scores = []

	for fold in range(cv):
		validation_data = train[num_validation_samples * fold:
							   num_validation_samples * (fold + 1)]
		target_validation_data = target[num_validation_samples * fold:
							   num_validation_samples * (fold + 1)]

		training_data = train[:num_validation_samples * cv].tolist() + \
			train[num_validation_samples * (cv + 1):].tolist()
		training_data = np.array(training_data)

		target_training_data = target[:num_validation_samples * cv] + \
			target[num_validation_samples * (cv + 1):]

		model = model_maker()
		model.fit(training_data, np.array(target_training_data), epochs=epochs,
				  batch_size=batch_size, validation_split=validation_split)

		validation_score = model.evaluate(validation_data,
										  np.array(target_validation_data))
		validation_scores.append(validation_score)

	return sum([el[0] for el in validation_scores]) / len(validation_scores), \
		sum([el[1] for el in validation_scores]) / len(validation_scores)


if __name__ == '__main__':
	read_tsv('corpus.tsv')
	print()
	# transform_tsv('corpus.tsv')