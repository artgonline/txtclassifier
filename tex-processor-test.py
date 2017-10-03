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
print(tp.get_chunks(r"NP: {<[CDJNP].*>+}", True)[0][0])
