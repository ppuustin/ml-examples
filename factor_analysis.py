import os, sys, time, re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from transformers import pipeline

class TransfSent():
    def __init__(self, sent_model, classes, device=0):
        # device=0 -> gpu, -1 -> cpu
        self.classifier = pipeline('sentiment-analysis', model=sent_model, device=device)
        self.classes = classes

    def classify(self, txt):
        result = self.classifier([txt])[0]
        s = result['label'].split(' ')[0]
        #score = round(result['score'], 5)
        return s

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}#{'PAD': self.PAD_token, 'SOS': self.SOS_token, 'EOS': self.EOS_token}
        self.index2word = {}#{self.PAD_token: 'PAD', self.SOS_token: 'SOS', self.EOS_token: 'EOS'}
        self.word2count = {}#{'PAD': 0, 'SOS': 0, 'EOS': 0}

        self.num_words = len(self.index2word.keys())
        self.num_sentences = 0
        self.longest_sentence = 0
        self.sent_lenghts = []

        self.char_uc  = '\x41-\x5a' # A-Z
        self.char_lc  = '\x61-\x7a' # a-z
        self.char_1uc = '\xc0-\xd4' # À-Ô
        self.char_ouc = '\xd6'      # Ö
        self.char_2uc = '\xd8-\xdf' # Ø-ß
        self.char_2lc = '\xe0-\xf6' # à-ö
        self.char_3lc = '\xf8-\xff' # ø-ÿ
        self.comma    = '\x2c'      # ,
        self.dot      = '\x2e'      # .
        self.space    = ' +'
        self.hyp      = r'[\x27]'   # '  60 `

        self.match = r'[^{}|^{}|^{}|^{}|^{}|^{}|^{}|^{}]'.format(self.char_uc, self.char_lc, 
            self.char_1uc, self.char_ouc, self.char_2uc, self.char_2lc, self.char_3lc, self.dot)    

    # ---------------------------------------------------------------

    def plot_table(self, df, fname, fs=(9,1), show=False):
        fig, ax = plt.subplots(1,1, figsize=fs)
        t = ax.table(cellText=df.values, colLabels=df.columns, loc='left', cellLoc='left')#, colWidths=[.1,.1,.8])
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.auto_set_column_width(col=list(range(len(df.columns))))
        plt.gca().axis('off')
        plt.tight_layout()
        if fname != None: plt.savefig(fname, dpi=190)
        if show: plt.show()
        plt.close()

    def plot_sent(self, fname, top_n=50, labels=False):
        unique = list(set(self.sent_lenghts))
        unique.sort(reverse=False)
        unique = unique[:top_n]
        #unique.reverse()   
        counts = [self.sent_lenghts.count(v) for v in unique]
        self.plot_dist(unique, counts, 'sentence stats', fname, labels)

    def plot_words(self, fname, top_n=50, labels=False):
        unique = sorted(self.word2count.items(), key=lambda x: x[1], reverse=True)   
        unique = unique[:top_n]
        unique = [v[0] for v in unique]
        unique.reverse()
        counts = [self.word2count[v] for v in unique]
        self.plot_dist(unique, counts, 'word stats', fname, labels)

    def plot_dist(self, unique, counts, title, fname=None, labels=False):
        fig = plt.figure(figsize=(5,7))
        bc = plt.barh(unique, counts)
        plt.title(title)
        #bc = plt.bar(range(len(unique)), counts)
        #plt.hist([self.sent_lenghts]) #, width=0.4, stacked=False, color = tgt_colors
        if labels: plt.bar_label(bc, counts, label_type='edge')
        plt.xticks(rotation=90)
        plt.tight_layout()
        if fname != None: plt.savefig(fname) # , dpi=190
        #plt.show()
        plt.close()

    # ---------------------------------------------------------------

    def dump_words(self, file=None, min_max=None, count=None, params='w'):
        words = self.get_words(min_max, count)
        with open(file, params, encoding='latin-1') as f: # "utf-8"
            for idx, line in enumerate(words):
                f.write(line)
                f.write('\n')

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    def cleanup(self, texts, lower=True):
        if lower: texts = [t.lower() for t in texts]
        texts = [re.sub(self.match, ' ', t) for t in texts]
        texts = [re.sub(self.hyp, '', t) for t in texts]
        return [re.sub(self.space, ' ', t) for t in texts]

    def get_words(self, min_max=None, count=None):
        words = []
        for index in range(self.num_words):
            word = self.to_word(index)
            if min_max and (len(word) < min_max[0] or len(word) > min_max[1]): continue
            if count and self.word2count[word] < count: continue
            words.append(word)
        words.sort() #key=str.lower
        return words

    def list_words(self, fname, min_max=None, count=None):
        words = self.get_words(min_max, count)
        for w in words: print(w)
        print('tot words:', len(words), '/', self.num_words)
        print('num_sentences:', self.num_sentences , 'longest_sentence:', self.longest_sentence)
        print('min_max:', min_max, 'count:', count)

        df_t = pd.DataFrame(columns=['', ' '])
        vals, vals[''] = {}, ''
        vals['tot words'] = f'{len(words)}/{self.num_words}'
        vals['num_sentences'] = str(self.num_sentences)
        vals['longest_sentence'] = str(self.longest_sentence)
        vals['min_max'] = str(min_max)
        vals['max_count'] = str(count)
        df_t = df_t.append(vals, ignore_index=True)
        self.plot_table(df_t, fname)

    def add_word(self, word):
        if word == '': return                                      # somewhere else!!!
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1 # Word exists; increase word count

    def add_sentence(self, sentence, sep=' '):
        sentence_len = 0
        for word in sentence.split(sep):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            self.longest_sentence = sentence_len
        self.num_sentences += 1
        self.sent_lenghts.append(sentence_len)

    def fit(self, texts, stopwords, clas=None, classes=None, cclass='c', sep=' ', min_len=2):
        print('fit')
        h, w = len(texts), self.num_words
        mtx = pd.DataFrame(np.zeros((h, w), dtype=np.int8))

        r = 0
        for i, text in enumerate(texts):
            if i % 100 == 0: print('row: {0}/{1}'.format(i, len(texts)))
            c = clas.classify(text)
            #print(c, text)
            clazz = classes[c]
            if pd.isna(text): continue
            for w in text.split(sep):
                w = re.sub(r'\W+', '', w)
                if len(w) <= min_len: continue 
                w = w.lower().strip()
                if w in stopwords: continue
                if w in self.word2index:
                    idx = self.to_index(w)
                    mtx.at[r,idx] = mtx.iat[r,idx] + 1
            mtx.at[r,-1] = clazz
            r += 1

        mtx.columns = list(self.word2index.keys()) + [cclass]
        return mtx.dropna().reset_index(drop=True)

class Factorizer():
    def __init__(self, n_components=2):
        self.n_components = n_components

    def do_fa(self, R, keys, file='output/pca_loadings_{0}.html'):
        #import seaborn as sns
        pca = PCA(self.n_components)
        x_pca = pca.fit_transform(R)
        print('explained_variance_ratio_', pca.explained_variance_ratio_)
        #print('Eigenvalues', pca.explained_variance_)
        #print('Eigenvectors', pca.components_)

        results = pd.DataFrame(pca.components_)
        #results.columns = ['_'+str(i)for i in range(R.shape[1])]
        results.columns = keys

        results = results.T
        #cm = sns.light_palette('red', as_cmap=True)
        styler = results.style.background_gradient(cmap='PuBu') # PuBu viridis
        styler.set_precision(3)
        with open(file,'w') as file:
            file.write(styler.render()) #styler.to_html() df.to_latex()

        #plt.show()
        #plt.close()

class Analyzer:
    def __init__(self, name):
        self.name = name
        self.voc = Vocabulary(name)

    def read_csv(self, file):
        #import chardet
        import pandas as pd        
        #print(enc['encoding'])

        if os.path.exists(file):
            with open(file, 'rb') as f:
                #enc = chardet.detect(f.read())
                return pd.read_csv(file, encoding='latin-1') # latin-1 UTF-8 ISO-8859-1 enc['encoding']
        else:
            return None

    def get_file_rows(self, file):
        rows = []
        with open(file, encoding='latin-1') as f: #utf-8
            for line in f: rows.append(line.strip())
        return rows

    def txt_to_sentences(self, txt, sep='.'):
        #return nl.tokenize.sent_tokenize(txt)
        sentences = txt.split(sep)
        return [s.strip() for s in sentences]

    def read_text(self, file):
        print(sys._getframe().f_code.co_name)
        rows = self.get_file_rows(file)

        all_sents = []
        for row in rows:
           sents = self.txt_to_sentences(row, sep='.')
           sents = [s for s in sents if s != '']
           sents = self.voc.cleanup(sents)
           for sent in sents:
               if len(sent) > 0: all_sents.append(sent)
               if len(sent) > 0: self.voc.add_sentence(sent)

        return self.voc, all_sents

    def voc_mtx(self):
        print(sys._getframe().f_code.co_name)
        props = {}
        props['text'] = 'kalevala'
        props['input_file'] = 'input/{0}.txt'.format(props['text'])
        props['word_stats'] = 'output/word_stats.png'
        props['voc_file'] = 'output/voc.txt'
        props['sent_stats'] = 'output/sent_stats.png'
        props['word_dist'] = 'output/word_dist.png'
        props['voc_fit'] = 'output/voc_fit.csv'
        props['min_max'] = (2,20)
        props['wcount'] = 1 
        props['smodel'] = None#'cardiffnlp/twitter-roberta-base-sentiment-latest'           # ???
        props['sclasses'] = {'negative':1, 'neutral':2, 'positive':3}
        props['sw_files'] = ['input/fi_stopwords.txt']
        self.stop_words = []
        for sw in props['sw_files']: self.stop_words.extend(self.get_file_rows(sw))                   
        cclass = '_class'

        mtx = self.read_csv(props['voc_fit'])

        if mtx is None:
            voc, all_sents = self.read_text(props['input_file'])
            voc.list_words(props['word_stats'], min_max=props['min_max'] , count=props['wcount']) #length=15, count=150
            voc.dump_words(file=props['voc_file'], min_max=props['min_max'] , count=props['wcount'])
            voc.plot_sent(props['sent_stats'], labels=True)
            voc.plot_words(props['word_dist'], labels=True)
            if props['smodel'] is not None:
                print('fit...')
                all_sents = all_sents[:500]                                               # <---------
                sent = TransfSent(props['smodel'], props['sclasses'])
                mtx = voc.fit(all_sents, self.stop_words, sent, sent.classes, cclass)
                mtx.to_csv(props['voc_fit'], index=False)

        print('factorize...')
        props['pca_file'] = 'output/pca_loadings_{0}.html'.format(props['text'])
        fac = Factorizer(n_components=5)
        fac.do_fa(mtx, mtx.columns, props['pca_file'])

        return mtx

if __name__ == '__main__':
    start = time.time()
    a = Analyzer('xxx')
    mtx = a.voc_mtx() # vcount mtx + sent
    end = time.time()
    print(f'Executed in {end - start:0.5f}s')
