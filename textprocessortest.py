from queue import Queue
import nltk
from textprocessor.textprocessor import TextProcessor

tp = TextProcessor.from_file(r'd:\repos\txtclassifier\textprocessor\data\news.txt')
# print(tp.tagged_sentences)
# print(tp.lwords)
# print(tp.lex_divercity())
# print(tp.word_freq('downloaded'))
# print(tp.word_freq('download', nltk.stem.lancaster.LancasterStemmer()))
# print(tp.word_freq('guarantee', lemmatise=True))
# print(tp.most_common_words(10))
# print(tp.tokenized_words)
# print(tp.get_similar('computer'))
# print(tp.get_word_context('for'))
# print(tp.get_collocations())
# print()
# print(tp.get_filtered_words(nostopwords=True))
# print(tp.get_chunks("NP: {<DT>?<JJ>*<NN>}", True)[0][0])
# print(tp.get_chunks(r"NP: {<[CDJNP].*>+}", True)[0][0])
# print(tp.get_np_chunks(True))
# tp.train_chunker(None, None)
# VERBS = ["VB","VBD","VBG","VBN", "VBP", "VBZ"]
# chunks = tp.get_np_chunks()
# svos = []
# for sent in chunks:
    # q = Queue()
    # s = []
    # svo = []
    # for i in range(len(sent)):
        # item = sent[i]
        # if (type(item) == nltk.tree.Tree and item.label() == 'NP') or item[1] in VERBS:
            # svo.append(item)
    # svos.append(svo)

# print(svos[0])
# print(tp.get_most_common_phrases(noun_tags=['NN']))

# train_data = nltk.corpus.treebank_chunk.chunked_sents()[:3200]
# test_data = nltk.corpus.treebank_chunk.chunked_sents()[3200:]
# tp.train_chunker(train_data, pickle_name='cnp_chunker_treebank.pickle', evaluate=True, test_data=test_data)

# tp_treebank = TextProcessor.from_file(r'd:\repos\txtclassifier\textprocessor\data\news.txt',
                                      # chunker='cnp_chunker_treebank.pickle')
# tp_treebank.get_np_chunks(evaluate=True)
# tp.get_np_chunks(evaluate=True)

data = list()
data.extend(nltk.corpus.conll2000.chunked_sents('train.txt', chunk_types=['NP']))
data.extend(nltk.corpus.conll2000.chunked_sents('test.txt', chunk_types=['NP']))
data.extend(nltk.corpus.treebank_chunk.chunked_sents())
print(len(data))
train_data = nltk.corpus.treebank_chunk.chunked_sents()[:13500]
test_data = nltk.corpus.treebank_chunk.chunked_sents()[13500:]
tp.train_chunker(train_data, pickle_name='cnp_chunker_comb.pickle', evaluate=True, test_data=test_data)
