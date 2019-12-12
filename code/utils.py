from typing import List, Tuple


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


def clean_tags(tags: List[str]) -> List[str]:
	n_tags = []
	tag: str
	for tag in tags:
		if tag != 'O' and tag.split('-')[1] != 'ORG':
			n_tags.append('O')
		else: n_tags.append(tag)
	return n_tags


class SentenceGetter:

	def __init__(self, **kwargs):
		if kwargs.get('words', False):
			self.text = self._to_sentences(kwargs['words'])
		else: raise Exception('incorrect arguments')

	@staticmethod
	def _to_sentences(sents: List[str]) -> List[List[str]]:
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


if __name__ == '__main__':
	read_tsv('corpus.tsv')
	print()
	# transform_tsv('corpus.tsv')