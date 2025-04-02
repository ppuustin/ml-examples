import os, re
#import chardet
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

# ----------------------------------------------------------------

class Cluster(object):
    def __init__(self, min_df, max_df, max_features):
       self.min_df = min_df
       self.max_df = max_df
       self.max_features = max_features

    def vectorize(self, sentences, n_clusters):
        vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, max_features=self.max_features)
        return vectorizer.fit_transform(sentences) # document-term matrix

    def kmeans(self, data, n_clusters=2):
        k = KMeans(n_clusters=n_clusters).fit(data)  
        cluster_centers = np.round(k.cluster_centers_).astype(int)
        return k.labels_, cluster_centers, k.inertia_

    def clust_to_file(self, labels, corpus, n_clusters, topic_file, write_lines, write_exel):
        clustered_sentences = [[] for i in range(n_clusters)]
        for sentence_id, cluster_id in enumerate(labels):
            clustered_sentences[cluster_id].append(sentence_id)

        to_exel = []
        for i, cluster in enumerate(clustered_sentences):
            lines = []
            for sid in cluster:
                lines.append(corpus[sid])
                to_exel.append({'txt':corpus[sid], 'cluster_id':i})

            if topic_file.endswith('.txt'):
                file = topic_file.format(n_clusters, i)
                write_lines(file, lines, 'w')

        if topic_file.endswith('.xlsx'):
            file = topic_file.format(n_clusters)
            write_exel(file, to_exel)

class Preprocessor():
    def __init__(self, stop_words, pipeline, min_len=1):
        self.min_len = min_len
        self.stop_words = stop_words
        self.char_uc  = '\x41-\x5a' # A-Z
        self.char_lc  = '\x61-\x7a' # a-z
        
        self.e1 = '\x80-\x90' # 128-144 # Ç-É 
        self.e2 = '\x93-\x9A' # 147-154 # ô-ø
        self.e3 = '\xA0-\xA5' # 160-165 # á-Ñ
        self.e4 = '\xB5-\xB7' # 181-183 # Á-À 
        self.e5 = '\xC6-\xC7' # 198-199 # ã-Ã 
        self.e6 = '\xD0-\xD8' # 208-216 # ð-Ï
        self.e7 = '\xE0-\xF6' # 224-237 # Ó-Ý   ???
        
        self.comma    = '\x2c'      # ,
        self.dot      = '\x2e'      # .
        self.space    = ' +'

        self.match = r'[^{}|^{}|^{}|^{}|^{}|^{}|^{}|^{}|^{}|^{}]'.format(
            self.char_uc, self.char_lc, 
            self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7,
            self.dot, self.comma)    

        self.special = 'RT |rt ' #[RT : |RT]
        self.mention = '@\S+ ?' #'@[A-Öa-ö0-9_:]+'
        self.hashtag = '#\S+ ?' #'#[A-Öa-ö0-9_:]+'
        self.urls = 'http[s]?://\S+'

        self.pipeline = pipeline

    def count_words(self, texts):
        all_words = []
        for text in texts:
            text = re.sub('[\W_]+', ' ', text) #^[\w-]+ 
            words = text.lower().split()
            all_words.extend(words)
        uniques = set(all_words)
        uniques = list(uniques)
        uniques.sort()
        return uniques, 'total number of unique words: {0}'.format(len(uniques))

    # ------------------------------------

    def cleanup(self, texts, lower=True, comma=False):
        #texts = [t.encode('latin-1') for t in texts]
        if lower: texts = [t.lower() for t in texts]
        texts = [re.sub(self.special, '', t) for t in texts]
        texts = [re.sub(self.mention, '', t) for t in texts]
        texts = [re.sub(self.hashtag, '', t) for t in texts]
        texts = [re.sub(self.urls, '', t) for t in texts]
        texts = [re.sub(self.match, ' ', t) for t in texts]
        texts = [re.sub(self.space, ' ', t) for t in texts]        
        return texts

    def remove_stopwords(self, txt, sep=' '):
        words = txt.split(sep)
        words = [w for w in words if w.lower().strip() not in self.stop_words]
        words = [w for w in words if len(w) > self.min_len]
        return ' '.join(words)

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        import spacy
        nlp = spacy.load(self.pipeline, disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = nlp(sent) 
            tokens = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            texts_out.append(' '.join(tokens))
        return texts_out       


class Classifier:
    def __init__(self, props):
        self.props = props

    def is_file(self, file):
        return os.path.isfile(file)

    def mk_dirs(self, file):
        dirs = os.path.dirname(file)
        if not os.path.isdir(dirs):
            os.makedirs(dirs)

    def write_exel(self, file, to_exel):
        self.mk_dirs(file)
        df = pd.DataFrame(to_exel)
        df.to_excel(file, index=False)
    
    def write_lines(self, file, lines, params):
        l = len(lines)
        print(f'writing {l} lines to {file}')
        self.mk_dirs(file)
    
        with open(file, params, encoding='latin-1') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

    def get_file_rows(self, file):
        rows = []    
        with open(file, encoding='latin-1') as f:
            for line in f: rows.append(line.strip())
        return rows	

    def read_csv(self, file):
        with open(file, 'rb') as f:
            #enc = chardet.detect(f.read())
            return pd.read_csv(file, encoding='latin-1')#enc['encoding']

    def read_file(self, file, trace=False):
        if not self.is_file(file): raise Exception(f'file: {file} not found.')
        try:
            df = self.read_csv(file)
        except Exception as e:
            rows = self.get_file_rows(file)
            df = pd.DataFrame(rows)
        return df

    def get_stopwords(self, input_dir):
        files = [f for f in os.listdir(input_dir) ]
        sws = []
        for f in files:
            if 'stopwords' in f:
                rows = self.get_file_rows(input_dir + f)
                sws.extend(rows)
        return sws

    def get_texts(self, file, column=None, trace=True):
        df = self.read_file(file, trace)
    
        if column == None or column == '':
            texts = df.iloc[:, 0].fillna('').values
        else:
            column = self.maby_int(df, column)
            df[column] = df[column].fillna('')
            texts = df[column].values
        return texts

    def preprocess(self, texts):
        sw = self.get_stopwords(self.props['in_dir'])
        pp = Preprocessor(sw, self.props['pipeline'], min_len=self.props['min_len'])
        _texts = pp.cleanup(texts, lower=True)
        #_texts = [pp.remove_stopwords(t, sep=' ') for t in _texts]
        #_texts = pp.lemmatization(_texts)
        uniques, line = pp.count_words(_texts)  

        return _texts, line

    def cluster(self, _texts, topics):
        clust = Cluster(self.props['min_df'], self.props['max_df'], self.props['max_features'])
        vectors = clust.vectorize(_texts, topics)
        labels, _, _ = clust.kmeans(vectors, n_clusters=topics)
        return labels, clust

    def run(self):
        topics = self.props['topics']
        print('read file:', self.props['file'])
        texts = self.get_texts(self.props['file'], column=self.props['column'] )

        print('preproc:', texts.shape)
        _texts, line = self.preprocess(texts)

        print('cluster:', len(_texts))
        labels, clust = self.cluster(_texts, topics)

        print('write:')
        clust.clust_to_file(labels, _texts, topics, self.props['clust_file'], self.write_lines, self.write_exel)

def main():
    props = {}
    props['file'] = 'input/kalevala.txt'
    props['column'] = None
    print(os.getcwd())
    props['min_len'] = 1
    props['in_dir'] = 'input/'
    props['out_dir'] = 'output/'
    #props['pipeline'] = './' + props['in_dir'] + 'fi_core_news_sm'
    props['pipeline'] = None #'fi_core_news_sm'    
    props['min_df'] = 1
    props['max_df'] = 0.95
    props['max_features'] = 1000    
    props['topics'] = 10  
    #props['clust_file'] = props['out_dir'] + 'topics_{0}/topics.xlsx'    
    props['clust_file'] = props['out_dir'] + 'topics_{0}/topic_{1}.txt'  
    
    c = Classifier(props)
    c.run() #TODO: just cluster example atm, impl classifier

if __name__ == '__main__':    
    main()