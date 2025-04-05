import os, requests

from datetime import datetime
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler    
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

from joblib import dump, load

import warnings
warnings.filterwarnings('ignore')

def dl_example(url, file):
    response = requests.get(url, stream=True)
    with open(file, 'wb') as handle:
        for data in tqdm(response.iter_content(chunk_size=1024), unit='kB'  ):
            handle.write(data)

def plot_pred(y_train, y_hats, y_test, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    plt.plot(y_train, 'g', marker='.')
    plt.plot(y_test, 'gray', marker='.')
    plt.plot(y_hats, 'orange' , marker='.')

    plt.xlabel('date')
    plt.ylabel(y_name)
    plt.legend(['train', 'actual', 'predicted'])
    plt.grid(True)
    plt.title(title)
    plt.show()

def plot_decomp(df, names, title='default title'):
    SMALL, MEDIUM, BIGGER = 8, 10, 12
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=False, figsize=[10, 6])

    plt.rc('legend', fontsize=MEDIUM)    # legend fontsize

    def plot_component(a, series, label):
        ax[a].plot(series, label=label, linewidth=1) 
        ax[a].set_ylim([series.min(), series.max()])
        ax[a].legend(loc='upper left')
        ax[a].tick_params(axis='both', which='minor', labelsize=MEDIUM) #major
        ax[a].grid()
        
    for i, n in enumerate(names):
        plot_component(i, df[n], n)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def printvals(df, groups, agg, y_name, dcol):
    var, var2, val, val2 = groups[0], groups[1], 'France', 'Furniture'
    _df = df.loc[(df[var]==val) & (df[var2]==val2)]
    agg = agg.lower()
    TREND, SEASONAL, RESIDUAL, ORIG = agg + '_trend', agg + '_seasonal', agg + '_residual', agg + '_original'
    if ORIG in _df.columns:
        print(_df[ORIG])
        print(_df[TREND])
    else:
        print(_df[y_name])
    #print(_df[SEASONAL])

    return _df

def decompose(original):
    s_deco = seasonal_decompose(x=original, model='additive') #, period=6
    _trend = s_deco.trend
    _seasonal = s_deco.seasonal
    _residual = s_deco.resid
    return _trend, _seasonal, _residual

def decomp(_df, y_name, dcol, agg='W'):
    original = _df[y_name]
    original = original.resample(agg).sum()

    _trend, _seasonal, _residual = decompose(original)

    agg = agg.lower()
    TREND, SEASONAL, RESIDUAL, ORIG = agg + '_trend', agg + '_seasonal', agg + '_residual', agg + '_original'
    df = pd.DataFrame()
    df[TREND] = _trend       #.astype(str)
    df[SEASONAL] = _seasonal #.astype(str)
    df[RESIDUAL] = _residual #.astype(str)
    df[ORIG] = original  #.astype(str)
    df[y_name] = original  #.astype(str)

    ignores = [dcol, y_name, TREND, SEASONAL, RESIDUAL, ORIG]
    for c in _df.columns:
        if c not in ignores: # '_' not in c or 
            df[c] = _df[c].iloc[0]

    df[dcol] = original.index.astype(str)
    return df

def do_decomp(df, agg, script=False):
    groups = ['Country', 'Category'] #Region
    dcol = 'Order Date'
    y_name = 'Sales'

    printvals(df, groups, agg, y_name, dcol)

    if script:
        df[dcol] = df[dcol].replace(r'Z', '') # .str.replace(r'Z', '')
    else:
        df[dcol] = df[dcol].str.replace(r'Z', '') # .str.replace(r'Z', '')

    df[dcol] = pd.to_datetime(df[dcol]) # format='%Y-%M-%dZ' '%d.%M.%Y'
    df = df.set_index(dcol)
    df = df.sort_values(by=dcol)
    #df = df.assign(w_trend=np.nan, w_seasonal=np.nan, w_residual=np.nan, w_original=np.nan)

    #df = df.groupby(groups).apply(decomp)
    by_group = df.groupby(groups)

    df = pd.DataFrame()
    start, end = 0, 0
    for state, frame in by_group:
        _df = decomp(frame, y_name, dcol, agg)
        end = end+len(_df.index)
        _df['Row ID'] = range(start, end)
        start = end
        df = _df if df.empty else df.append(_df)

    printvals(df, groups, agg, y_name, dcol)
    return df

# ------------------------------------------------------------------

def get_train_test(df, year=2017):
    #df.sort_index(inplace=True)
    train = df[df.index < datetime(year=year, month=5, day=1)]
    test = df[df.index >= datetime(year=year, month=5, day=1)]
    return train, test

def split_data(train, test, x_names, y_name):
    #x_train, x_test, y_train, y_test = train_test_split(inputs, output, random_state=42) # df[x_names] df[y_name] 
    x_train, y_train = train[x_names], train[y_name]
    x_test, y_test = test[x_names], test[y_name]
    #x_train.sort_index(inplace=True)
    #y_train.sort_index(inplace=True)
    #x_test.sort_index(inplace=True)
    #y_test.sort_index(inplace=True)
    return x_train, y_train, x_test, y_test 

def remove_outlier(df, col):
    q = df[col].quantile([0.25,0.5,0.75])
    IQR = q[0.75] - q[0.25]
    lower = q[0.25] - 1.5 * IQR # q[0.5]
    upper = q[0.75] + 1.5 * IQR # q[0.5]
    return df[(df[col] >= lower) & (df[col] <= upper)]

def round_y(y, nearest=100):
    return round(y/nearest)*nearest

def prepocess(df, dcol, script=False):
    df = remove_outlier(df, y_name)
    df[y_name] = df[y_name].apply(round_y)

    if script:
        df[dcol] = df[dcol].replace(r'Z', '') # .str.replace(r'Z', '')
    else:
        df[dcol] = df[dcol].str.replace(r'Z', '') # .str.replace(r'Z', '')
    
    df[dcol] = pd.to_datetime(df[dcol])

    df['Day'] = df[dcol].dt.day
    df['Month'] = df[dcol].dt.month

    df = df.set_index(dcol)
    df = df.sort_values(by=dcol)

    return df

def do_explore(df, m_name='grb'):
    import numpy as np
    from sklearn import metrics

    df = prepocess(df, dcol=dcol, script=True)

    df_train, df_test = get_train_test(df)
    x_train, y_train, x_test, y_test = split_data(df_train, df_test, x_names, y_name)
    print('train/x/y:', x_train.shape, y_train.shape)
    print('test/x/y:', x_test.shape, y_test.shape)
    
    num_cols = selector(dtype_exclude=object)
    cat_cols = selector(dtype_include=object)
    num_col = num_cols(x_train)
    cat_col = cat_cols(x_train)    

    ohe = OneHotEncoder(handle_unknown="ignore")
    std = StandardScaler()
    preproc = ColumnTransformer([('one-hot-encoder', ohe, cat_col),('standard_scaler', std, num_col)])
    if m_name=='svr': method = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    if m_name=='lin': method = LinearRegression()    
    if m_name=='grb': method = GradientBoostingRegressor(random_state=42)

    model = make_pipeline(preproc, method)
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    msq = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)
    print(m_name, 'rmse:', msq)

def test_model(df):
    var, var2 = x_names[-2], x_names[-1]
    #val, val2 = 'Spain', 'Technology'          # <---
    val, val2 = 'Sweden', 'Technology'
    #val, val2 = 'France', 'Technology'

    df = df.loc[(df[var]==val) & (df[var2]==val2)]
    df_train, df_test = get_train_test(df)

    title = '{0} = {1}, {2} = {3}'.format(var, val, var2, val2)
    plot_pred(df_train[y_name], df_test[y_hat], df_test[y_name], title) #train pred gt

def inference(df, script=False):
    print('** ENTER **', 'df.shape:', df.shape)

    df = prepocess(df, dcol=dcol, script=script)
    idx = df.index
    #df[y_hat] = None

    df_train, df_test = get_train_test(df)
    _, _, x_test, _ = split_data(df_train, df_test, x_names, y_name)
    print('train:', df_train.shape, 'test:', df_test.shape)
    print('load:', model_name)
    
    model = load(model_name)
    y_pred = model.predict(df_test[x_names])
    df_test[y_hat] = y_pred

    _df = df_train.append(df_test)
    _df[y_hat] = _df[y_hat].astype(float) # str float
    _df[dcol] = idx.astype(str)

    print('** EXIT **', '_df.shape:', _df.shape)
    return _df

def do_inference(df):
    df = inference(df, script=True)
    print(df.columns, '\n', df.dtypes)
    test_model(df)

def train(x_train, y_train):
    num_cols = selector(dtype_exclude=object)
    cat_cols = selector(dtype_include=object)
    num_col = num_cols(x_train)
    cat_col = cat_cols(x_train)    

    ohe = OneHotEncoder(handle_unknown="ignore")
    std = StandardScaler()
    preproc = ColumnTransformer([('one-hot-encoder', ohe, cat_col),('standard_scaler', std, num_col)])

    if 'svr' in model_name: method = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    if 'lin' in model_name: method = LinearRegression()    
    if 'grb' in model_name: method = GradientBoostingRegressor(random_state=42)

    model = make_pipeline(preproc, method)
    return model.fit(x_train, y_train)

def do_train(df, save=False):
    df = prepocess(df, dcol=dcol, script=True)

    df_train, df_test = get_train_test(df)
    x_train, y_train, x_test, y_test = split_data(df_train, df_test, x_names, y_name)

    model = train(x_train, y_train)
    if save: dump(model, model_name)

# ------------------------------------------------------------------

def decomps(df):
    agg = 'M' # W M
    df = do_decomp(df, agg, script=True)
   
    groups = ['Country', 'Category'] #Region
    var, var2, val, val2 = groups[0], groups[1], 'France', 'Furniture'
    var, var2, val, val2 = groups[0], groups[1], 'France', 'Technology'
    #var, var2, val, val2 = groups[0], groups[1], 'Spain', 'Technology'    
    #var, var2, val, val2 = groups[0], groups[1], 'Netherlands', 'Furniture'
    #var, var2, val, val2 = groups[0], groups[1], 'Austria', 'Furniture'
    
    title = val+' / '+val2
    df = df.loc[(df[var]==val) & (df[var2]==val2)]
    agg = agg.lower()
    names = [agg + '_original', agg + '_trend', agg + '_seasonal', agg + '_residual']
    #print(df.columns)
    #print(df.dtypes)
    plot_decomp(df, names, title=title)

if __name__ == '__main__':
    file = 'Sample%20-%20EU%20Superstore.xls'
    repo = 'https://github.com/PacktPublishing/Building-Interactive-Dashboards-with-Tableau-10.5'
    url = repo + '/raw/refs/heads/master/' + file
    input_file = 'input/' + file
    #dl_example(url, input_file)    
    df = pd.read_excel(input_file)

    #model_name = 'output/svr.joblib' #../
    #model_name = 'output/lin.joblib'
    model_name = 'output/grb.joblib'

    #x_names = ['Category']
    x_names = ['Day', 'Month', 'Quantity', 'Country', 'Category'] #Region
    #x_names = ['Country/Region', 'Category']

    dcol = 'Order Date'
    y_name = 'Sales'
    y_hat = 'y_hat'

    decomps(df) # split to components

    #for m in ['svr', 'lin', 'grb']: do_explore(df, m)
    #do_train(df, save=True)
    #do_inference(df)