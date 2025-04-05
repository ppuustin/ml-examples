import os, os.path, time, random, math, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import threading, itertools

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler    
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn import metrics

import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense  
from tensorflow.keras.models import Model

class IndexGenerator(object):
    def __init__(self, ids):
        self.ids = ids
        self.lock = threading.Lock()
        self.it = itertools.cycle(self.ids)

    def shuffle(self):
        np.random.shuffle(self.ids)            
        self.it = itertools.cycle(self.ids) 
     
    def __len__(self):
        return len(self.ids)          

    def __next__(self):
        with self.lock:
            return next(self.it) #next index

class SampleGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        ids =  list(range(0, x.shape[0]))
        self.id_gen = IndexGenerator(ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x
        self.y = y
        self.cval = 'value' # TODO the name?

    def get_xy(self):
        x = [self.x.iloc[i][self.cval] for i in self.id_gen.ids]
        y = [self.y.iloc[i] for i in self.id_gen.ids]        
        x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
        return x, y

    # -------------------------------------------------------------------
    # from tf.keras.utils.Sequence

    def __len__(self):
        n = len(self.id_gen)/self.batch_size
        return int(np.floor(n))              # batches per epoch

    def __getitem__(self, index):
        return self.__get_batch(index)

    def on_epoch_end(self):
        if self.shuffle == True:       
            self.id_gen.shuffle() 

    # -------------------------------------------------------------------
    # private methods

    def __get_batch(self, index):
        '''dataset and task specific implementation'''
        x_seqs, y_seqs = [], []
        for b in range(self.batch_size):
            sample_id = next(self.id_gen)
            #if line == None: raise Exception(f"id {sample_id} not found")
            x = self.x.iloc[sample_id][self.cval] # TODO the name?
            y = self.y.iloc[sample_id]          
            x_seqs.append(x)
            y_seqs.append(y)

        x_seqs = np.array(x_seqs, dtype=np.float32) #float16 .astype(int)
        y_seqs = np.array(y_seqs, dtype=np.float32).astype(int) #np.array([], dtype=np.float32)
        return x_seqs, y_seqs

class LossCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, epochs):
        self.epochs = epochs
        self.count = 0
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.maes = []
        self.val_maes = []
        self.epoch_time = []
        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.maes.append(logs.get('mae'))
        self.val_maes.append(logs.get('val_mae'))
        self.epoch_time.append(time.time() - self.epoch_start_time)
        self.count += 1
        if self.count == self.epochs:
            self.do_plot()

    def get_losses(self):
        return self.losses, self.val_losses

    def do_plot(self):
        #clear_output(wait=True)
        self.plot_it('losses', 'epoch', 'loss', self.losses, self.val_losses)
        self.plot_it('err', 'epoch', 'mae', self.maes, self.val_maes)
        self.plot_it('epoch time', 'epoch', 'time (s)', self.epoch_time)

    def plot_it(self, title, xlab, ylab, what, val=None):
        plt.figure(figsize=(10,5)) # <w,h>
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.plot(what, label=ylab)
        if val != None: plt.plot(val, label='val_'+ylab)
        plt.legend(loc='best')
        plt.grid(which='both')
        plt.show()
        plt.close()
       
def get_data(file):
    print(sys._getframe().f_code.co_name)
    df = pd.read_csv(file, encoding='latin-1')
    print(df.columns)
    if 'sine' in file:
        dcol = 'date'
        y_name = 'value'
        x_names = [y_name]
        #pd.to_numeric(df[y_name])
        start, end, freq = '1-1-2000', '', 'D'
        #df[dcol] = pd.date_range(start=start, freq=freq, periods=df.shape[0])
        
        #for index, row in df.iterrows():
        #    print('{0},{1}'.format(str(row[dcol]).split(' ')[0], row['value']))
        df[dcol] = pd.to_datetime(df[dcol], format='%Y-%m-%d')
        df = df.set_index(dcol)
        df = df.sort_values(by=dcol)
    else:
        print('nothing')
    
    return df, x_names, y_name
    
def get_train_test(df, pct=0.8):
    split = int(df.shape[0] * pct)
    return df[:split], df[split:]

def split_data(train, test, x_names, y_name, agg=None):
    x_train, y_train = train[x_names], train[y_name]
    x_test, y_test = test[x_names], test[y_name]

    if agg != None:
        x_train = x_train.resample(agg).sum()
        y_train = y_train.resample(agg).sum()#.to_frame()
        x_test = x_test.resample(agg).sum()
        y_test = y_test.resample(agg).sum()#.to_frame()

    x_train, y_train = x_train.iloc[:-1], y_train.iloc[1:]
    x_test, y_test = x_test.iloc[:-1], y_test.iloc[1:]

    return x_train, y_train, x_test, y_test   

def get_generators(df, batch_size, x_names, y_name, pct_train=0.8, pct_test=0.2, pct_val=0.2):
    split = int(df.shape[0] * pct_train)
    df_train, df_test = df[:split], df[split:]

    split = int(df_train.shape[0] * (1 - pct_val))
    df_train, df_val = df_train[:split], df_train[split:]
    print('train/test/val:', df_train.shape, df_test.shape, df_val.shape)
    
    x_train, y_train = df_train[x_names], df_train[y_name]
    x_test, y_test = df_test[x_names], df_test[y_name]    
    x_val, y_val = df_val[x_names], df_val[y_name]     
    
    x_train, y_train = x_train.iloc[:-1], y_train.iloc[1:]
    x_test, y_test = x_test.iloc[:-1], y_test.iloc[1:]
    x_val, y_val = x_val.iloc[:-1], y_val.iloc[1:]

    train_gen = SampleGenerator(x_train, y_train, batch_size)
    test_gen = SampleGenerator(x_test, y_test, batch_size)
    val_gen = SampleGenerator(x_val, y_val, batch_size)
    return train_gen, test_gen, val_gen
# --------------------------------------------------------------

def train_sk(x_train, y_train):
    print(sys._getframe().f_code.co_name)
    num_cols = selector(dtype_exclude=object)
    cat_cols = selector(dtype_include=object)
    num_col = num_cols(x_train)
    cat_col = cat_cols(x_train)    

    ohe = OneHotEncoder(handle_unknown="ignore")
    std = StandardScaler()
    preproc = ColumnTransformer([('one-hot-encoder', ohe, cat_col),('standard_scaler', std, num_col)])
    method = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)

    model = make_pipeline(preproc, method)
    return model.fit(x_train, y_train)

def train_tf(x_train, y_train):
    print(sys._getframe().f_code.co_name)
    input_length = 1
    input_dim = 1
    lstm_units_1 = 2**6
    kernel_act = 'linear' # relu softmax
    loss = 'mean_squared_error'
    optimizer = 'adam'
    metrics=['mae']

    batch_size = 2**3
    epochs = 5
    verbose = 1 
    validation_split = 0.2
    callbacks = [LossCallback(epochs)] 

    input_layer = Input(shape=(input_length,input_dim))
    x = LSTM(lstm_units_1)(input_layer)
    output = Dense(input_dim, activation=kernel_act)(x) # , 
    model = Model([input_layer], output)          

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)   

    model.fit(x_train, y_train, 
              batch_size=batch_size, epochs=epochs, 
              verbose=verbose, validation_split=validation_split,
              callbacks=callbacks)  
    return model

def train_gen(train_gen, val_gen, test_gen):
    print(sys._getframe().f_code.co_name)
    input_length, input_dim = 1, 1
    lstm_units_1, kernel_act = 2**6, 'linear'
    loss, optimizer, metrics = 'mean_squared_error', 'adam',  ['mae']
    epochs = 5

    callbacks = [LossCallback(epochs)] 

    input_layer = Input(shape=(input_length,input_dim))
    x = LSTM(lstm_units_1)(input_layer)
    output = Dense(input_dim, activation=kernel_act)(x) # , 
    model = Model([input_layer], output)          

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)   

    history = model.fit_generator(
                    generator=train_gen,
                    validation_data=val_gen,                     
                    callbacks=callbacks,
                    epochs=epochs, 
                    shuffle=True)
    return model

def inference(model, x_test):
    print('inference :', x_test.shape)
    return model.predict(x_test)

def plot_pred(y_train, y_test, y_hat, y_name, title):
    plt.figure(figsize=(10,5))
    plt.plot(y_train, 'g', marker='.')
    plt.plot(y_test, 'gray', marker='.')
    plt.plot(y_hat, 'orange' , marker='.')

    plt.xlabel('date')
    plt.ylabel(y_name)
    plt.legend(['train', 'actual', 'predicted'])
    plt.title(title)
    plt.grid(which='both')
    plt.show()
    plt.close()

def main_gen(file):
    df, x_names, y_name = get_data(file)
    tra_gen, tes_gen, val_gen = get_generators(df, 2**3, x_names, y_name)
    model = train_gen(tra_gen, tes_gen, val_gen)
    x_test, y_test = tes_gen.get_xy()
    y_hat = inference(model, x_test)
    msq = round(np.sqrt(metrics.mean_squared_error(y_test, y_hat)), 2)
    print('msq:', msq) #TODO, not ok yet
    
def main(file):
    df, x_names, y_name = get_data(file)
    print('in:', df.shape)
    agg = None
    df_train, df_test = get_train_test(df)    
    x_train, y_train, x_test, y_test = split_data(df_train, df_test, x_names, y_name, agg=agg)
    
    print('train/test:', df_train.shape, df_test.shape)
    print('xy/xy:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    #model = train_sk(x_train, y_train)
    model = train_tf(x_train, y_train)

    y_hat = inference(model, x_test)
    print('hat/test/train:',  y_hat.shape, y_test.shape, y_train.shape)

    msq = round(np.sqrt(metrics.mean_squared_error(y_test, y_hat)), 2)
    print('msq:', msq)
 
    print('x{0}y{1} rmse'.format(' '*12, ' '*12))
    print('{:13s}{:13s}{}'.format(','.join(x_names), y_name, msq))
    
    df_test = df_test.iloc[1:]
    df_test['y_hat'] = y_hat
    _df = df_train.append(df_test)

    title = 'y: {0}, agg: {1}, rmse: {2}'.format(y_name, agg, msq)
    plot_pred(y_train, y_test, _df['y_hat'], y_name, title)

if __name__ == '__main__':
    #pd.set_option('display.max_rows', 6000)
    #pd.set_option('display.max_columns', 500)
    #pd.set_option('display.width', 1000)
    print(os.getcwd())
    file = os.getcwd() + '/input/sine.csv' # sine-wave

    main(file)
    #main_gen(file)

'''

The conflict is caused by:
    The user requested numpy
    pandas 1.5.3 depends on numpy>=1.20.3; python_version < "3.10"
    scikit-learn 1.0.2 depends on numpy>=1.14.6
    tensorflow 2.5.0 depends on numpy~=1.19.2

'''