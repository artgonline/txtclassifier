import nltk
import os
import pickle


class ConsecutiveNPChunkTagger(nltk.TaggerI):
    '''
    Create IOB tagget text based on POS tagged input data.
    Uses MaxentClassifier model. Extracts a number of features from the POS tagged text,
    Inculding:
        previous word tag
        next word tag
        different combinations of previos and current tags (see function doc string for more detail.)
    '''
    def __init__(self, train_sents=None, pickle_name=None, save=False):
        '''Trains new tagger model or loads existing one.''' 
        self._pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pickles', pickle_name)
        if train_sents == None:
            # Load exisiting model
            with open(self._pickle_path, 'rb') as f:
                self.classifier = pickle.load(f)
        else:
            train_set = []
            for tagged_sent in train_sents:
                untagged_sent = nltk.tag.untag(tagged_sent)
                history = []
                for i, (word, t) in enumerate(tagged_sent):
                    featureset = self.npchunk_features(untagged_sent, i, history)
                    train_set.append( (featureset, t) )
                    history.append(t)
            nltk.config_megam(r'c:\progs\megam\megam.exe')
            self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)
            if save:
                # save newly trained model with name specified
                with open(self._pickle_path, 'wb') as f:
                    pickle.dump(self.classifier, f)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = self.npchunk_features(sentence, i, history)
            t = self.classifier.classify(featureset)
            history.append(t)
        return zip(sentence, history)

    def npchunk_features(self, sentence, i, history):
        '''
        Extracts features from POS tagged sentence.
        Returns a dictionary of the following features:
            pos             current word POS tag
            word            current word
            prevpos         previous word POS tag
            nextpos         next word POS tag
            prevpos+pos     previous and current words POS  tags combined
            pos+nextpos     current and next words POS tags combined
            tags-since-dt   all tags since last determiner or beggining of sentence. 
        '''
        try:
            word, pos = sentence[i]
        except ValueError:
            print('FAILED SENT:', sentence[i])
        if i == 0:
            prevword, prevpos = "<START>", "<START>"
        else:
            prevword, prevpos = sentence[i-1]

        if i == len(sentence)-1:
            nextword, nextpos = "<END>", "<END>"
        else:
            nextword, nextpos = sentence[i+1]
        return {"pos": pos,
                "word": word,
                "prevpos": prevpos,
                "nextpos": nextpos,
                "prevpos+pos": "%s+%s" % (prevpos, pos),
                "pos+nextpos": "%s+%s" % (pos, nextpos),
                "tags-since-dt": self.tags_since_dt(sentence, i)}

    def tags_since_dt(self, sentence, i):
        tags = set()
        for word, pos in sentence[:i]:
            if pos == 'DT':
                tags = set()
            else:
                tags.add(pos)
        return '+'.join(sorted(tags))


class ConsecutiveNPChunker(nltk.ChunkParserI):

    def __init__(self, train_sents=None, pickle_name='cnp_chunker.pickle', save=False):
        '''Trains new tagger model or loads saved tagger and returns NP chunker'''
        tagged_sents = None
        if train_sents != None:
            tagged_sents = [[((w,t),c) for (w,t,c) in
                             nltk.chunk.tree2conlltags(sent)]
                            for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents, pickle_name, save)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)
