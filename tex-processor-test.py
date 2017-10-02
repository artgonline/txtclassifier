import nltk
from textprocessor.textprocessor import TextProcessor

tp = TextProcessor.from_file(r'd:\repos\txtclassifier\textprocessor\data\news.txt')
# print(len(tp.sentences))
# print(tp.lwords)
# print(tp.lex_divercity())
# print(tp.word_freq('downloaded'))
# print(tp.word_freq('download', nltk.stem.lancaster.LancasterStemmer()))
# print(tp.word_freq('guarantee', lemmatise=True))
# print(tp.most_common_words(10))
# print(tp.tokenized_words)
# print(tp.get_similar('computer'))
# print(tp.get_word_context('for'))
print(tp.get_collocations())
print()
print(tp.get_filtered_words(nostopwords=True))
