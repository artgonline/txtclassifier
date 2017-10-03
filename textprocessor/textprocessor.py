import nltk

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

    def __init__(self, s):
        self._raw_text = s
        # this is a pre-traned Punkt sentence tokenizer (see http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.punkt)
        self.sentences = nltk.sent_tokenize(self._raw_text)
        self.sentences = [nltk.word_tokenize(w.lower()) for w in self.sentences]
        self.tagged_sentences = [nltk.pos_tag(s) for s in self.sentences]
        self.lwords = []
        for s in self.sentences:
            self.lwords.extend([w for w in s])
        
        self._fdist = None
        self.text = nltk.Text(self.lwords)

    def from_file(f_path):
        with open(f_path) as f:
            return TextProcessor(f.read())

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

    def lex_divercity(self,):
        '''Lexical richness of the text, i.e. how many new words are used.'''
        if self.lwords == None:
            return 0
        return len(set(self.lwords)) / len(self.lwords)

    def word_freq(self, w, stemmer=None, lemmatise=False):
        '''
        Returns the frequency of given word.
        If stemmer argument is provided, the stemmer will be applied to words and
        frequency distribution will be calculated based on the stemmer output.
        '''
        if self.lwords == None:
            return 0

        words = None
        if stemmer != None and isinstance(stemmer, nltk.stem.api.StemmerI):
            words = [stemmer.stem(word) for word in self.lwords]
        if lemmatise:
            lemmatiser = nltk.stem.WordNetLemmatizer()
            if words == None:
                words = self.lwords[:]
            words = [lemmatiser.lemmatize(word) for word in words]

        fdist = self._get_words_freq_dist(words)
        return fdist[w]

    def most_common_words(self, n=None):
        '''Wrapper for nltk.FreqDist.most_common'''
        if self.lwords == None:
            return 0

        fd = self._get_words_freq_dist()
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

    def get_filtered_words(self, nostopwords=False, lang='english'):
        if self.lwords == None:
            return []

        words = self.lwords[:]

        if nostopwords:
            stopwords = set(nltk.corpus.stopwords.words(lang))
            words = [w for w in words if w not in stopwords]

        return words

    def get_chunks(self, grammar, evaluate=False):
        cp = nltk.RegexpParser(grammar)
        if evaluate:
            test_data = nltk.corpus.conll2000.chunked_sents('test.txt', chunk_types=['NP'])
            print('EVALUTATION:', cp.evaluate(test_data))
        chunks = []
        for s in self.tagged_sentences:
            chunks.append(cp.parse(s))

        return chunks
