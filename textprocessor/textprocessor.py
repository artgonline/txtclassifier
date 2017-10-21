from collections import Counter
import nltk
import string

# from . import conchunker
import conchunker

class TextProcessor:
    '''
    Text processor class.

    tp = TextProcessor(string)
    tp = TextProcessor.from_file(file_path)

    Words tags:
        ADJ 	adjective 	            new, good, high, special, big, local
        ADP 	adposition 	            on, of, at, with, by, into, under
        ADV 	adverb 	                really, already, still, early, now
        CONJ 	conjunction 	        and, or, but, if, while, although
        DET 	determiner, article 	the, a, some, most, every, no, which
        NOUN 	noun 	                year, home, costs, time, Africa
        NUM 	numeral 	            twenty-four, fourth, 1991, 14:24
        PRT 	particle 	            at, on, out, over per, that, up, with
        PRON 	pronoun 	            he, their, her, its, my, I, us
        VERB 	verb 	                is, say, told, given, playing, would
        . 	    punctuation marks 	    . , ; !
        X 	    other 	                ersatz, esprit, dunno, gr8, univeristy
    '''

    def __init__(self, s, chunker='cnp_chunker.pickle'):
        self._raw_text = s
        self.chunker_name = chunker
        # this is a pre-traned Punkt sentence tokenizer (see http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.punkt)
        self.sentences = nltk.sent_tokenize(self._raw_text)
        self.sentences = [nltk.word_tokenize(w.lower()) for w in self.sentences]
        self.tagged_sentences = [nltk.pos_tag(s) for s in self.sentences]
        self.lwords = []
        for s in self.sentences:
            self.lwords.extend([w for w in s])
        
        self._fdist = None
        self.text = nltk.Text(self.lwords)

    def from_file(f_path, chunker='cnp_chunker.pickle'):
        with open(f_path, encoding='utf8') as f:
            return TextProcessor(f.read(), chunker)

    def _get_np_phrases(self,):
        get_np_trees = lambda tree: list(tree.subtrees(filter=lambda t: t.label() == 'NP'))
        np_trees = [t for tree in self.get_np_chunks() for t in get_np_trees(tree)]
        return TextProcessor.generate_phrases_from_trees(np_trees)

    def _get_most_commont_nouns(self, noun_tags, n=10):
        nouns = [w[0] for s in self.tagged_sentences for w in s if w[1] in noun_tags]
        min_freq = min([w[1] for w in self.most_common_words(words=nouns, n=n)])
        return [w[0] for w in self.most_common_words(words=nouns) if w[1] >= min_freq]

    def _get_words_freq_dist(self, wlist=None):
        '''
        Returns FreqDist object for provided list of words,
        or for text words.
        '''
        if wlist != None:
            return nltk.FreqDist(wlist)

        if self._fdist == None:
            self._fdist = nltk.FreqDist(self.lwords)
        return self._fdist

    @staticmethod
    def generate_phrases_from_trees(trees, excl_tags=['POS', 'DT', 'PRP$']):
        phs = set()
        for t in trees:
            leaves = [l[0] for l in t.leaves() 
                      if not any([c in string.punctuation for c in l[1]]) 
                          and l[1] not in excl_tags]
            # yield ' '.join(leaves)
            phs.add(' '.join(leaves))
        return phs

    def lex_divercity(self,):
        '''Lexical richness of the text, i.e. how many new words are used.'''
        if self.lwords == None:
            return 0
        return len(set(self.lwords)) / len(self.lwords)

    def word_freq(self, w, words=None, stemmer=None, lemmatise=False):
        '''
        Returns the frequency of given word.
        If stemmer argument is provided, the stemmer will be applied to words and
        frequency distribution will be calculated based on the stemmer output.
        '''
        if words == None:
            if self.lwords == None:
                return 0

            words = self.lwords[:]

        if stemmer != None and isinstance(stemmer, nltk.stem.api.StemmerI):
            words = [stemmer.stem(word) for word in words]
        if lemmatise:
            lemmatiser = nltk.stem.WordNetLemmatizer()
            words = [lemmatiser.lemmatize(word) for word in words]

        fdist = self._get_words_freq_dist(words)
        return fdist[w]

    def most_common_words(self, words=None, n=None):
        '''Wrapper for nltk.FreqDist.most_common'''
        if words == None:
            if self.lwords == None:
                return 0
            words = self.lwords

        fd = self._get_words_freq_dist(words)
        if n == None:
            n = len(self.lwords)
        return fd.most_common(n)

    def get_similar(self, w):
        '''Returns words that appear in the same context as w'''
        if self.text == None or w == None:
            return []
        return self.text.similar(w)

    def get_word_context(self, w):
        '''
        Wrapper for nltk Text.concordance function.
        Returns the context surrounding the given word.
        '''
        if self.text == None:
            return []

        return self.text.concordance(w)

    def get_collocations(self,):
        '''Wrapper for nltk text.collocations()'''
        if self.text == None:
            return []
        return self.text.collocations()

    def get_filtered_words(self, nostopwords=False, nopunct=False,
                           stemm=False, lemmatize=False, lang='english'):
        if self.lwords == None:
            return []

        words = self.lwords[:]

        if lemmatize:
            # Apply WordNetLemmatizer first
            wnl = nltk.WordNetLemmatizer()
            words = [wnl.lemmatize(w) for w in words]

        if stemm:
            # Apply PorterStemmer
            stemmer = nltk.PorterStemmer()
            words = [stemmer.stem(w) for w in words]

        if nostopwords:
            stopwords = set(nltk.corpus.stopwords.words(lang))
            words = [w for w in words if w not in stopwords]

        if nopunct:
            words = [w for w in words if all(c not in string.punctuation for c in w)]

        return words

    def get_chunks_for_grammar(self, grammar, evaluate=False):
        cp = nltk.RegexpParser(grammar)
        if evaluate:
            test_data = nltk.corpus.conll2000.chunked_sents('test.txt', chunk_types=['NP'])
            print('EVALUTATION:', cp.evaluate(test_data))
        chunks = []
        for s in self.tagged_sentences:
            chunks.append(cp.parse(s))

        return chunks

    def train_chunker(self, train_sents, pickle_name='cnp_chunker.pickle', evaluate=True, test_data=None):
        if train_sents == None:
            train_sents = nltk.corpus.conll2000.chunked_sents('train.txt', chunk_types=['NP'])
        chunker = conchunker.ConsecutiveNPChunker(train_sents, pickle_name=pickle_name, save=True)

        if evaluate:
            if test_data == None:
                test_data = nltk.corpus.conll2000.chunked_sents('test.txt', chunk_types=['NP'])
            print('EVALUTATION:', chunker.evaluate(test_data))


    def get_np_chunks(self, evaluate=False):
        if self.tagged_sentences == None:
            return None

        chunker = conchunker.ConsecutiveNPChunker(pickle_name=self.chunker_name)

        if evaluate:
            test_data = nltk.corpus.conll2000.chunked_sents('test.txt', chunk_types=['NP'])
            print('EVALUTATION:', chunker.evaluate(test_data))

        return [chunker.parse(s) for s in self.tagged_sentences]

    def get_most_common_phrases(self, n=10, noun_tags = ['NN', 'NNS']):
        '''
        Extracts NP phrases from the text and return the most reaccuring 
        phrases and their frequency.
        '''
        # Get all noun phrases from the text
        phrases = self._get_np_phrases()
        # Get most used nouns
        most_common_nouns = self._get_most_commont_nouns(noun_tags)

        cnt = Counter()
        for nn in most_common_nouns:
            for p in phrases:
                if nn in p:
                    cnt[p] += 1
        return cnt.most_common(n)
