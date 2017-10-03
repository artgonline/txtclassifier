import nltk
from conchunker import ConsecutiveNPChunker

test_sents = nltk.corpus.conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = nltk.corpus.conll2000.chunked_sents('train.txt', chunk_types=['NP'])
chunker = ConsecutiveNPChunker(train_sents)
