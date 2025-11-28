import os, os.path, time, random, math, sys
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

# ------------------------------------------------
# Utility functions

def plot_evr(data, n_components=2):
    pca = PCA(n_components)
    x_pca = pca.fit_transform(data)
    #print('explained_variance_ratio_', pca.explained_variance_ratio_)
    #print('eigenvalues', pca.explained_variance_)
    #print('eigenvectors', pca.components_)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))    
    axes[0].plot(pca.explained_variance_ratio_)    
    axes[0].set_title('explained variance ratio')
    axes[0].set_xlabel('principal components')
    axes[0].set_ylabel('explained variance')
    axes[1].plot(np.cumsum(pca.explained_variance_ratio_))
    axes[1].set_title('cumsum')
    axes[1].set_xlabel('number of components')
    axes[1].set_ylabel('cumulative explained variance')
    plt.show()
    
def plot_loadings(data, n_components=2, file=None):
    #file='output/pca_loadings_{0}.html'
    #file = file.format('xx')
    pca = PCA(n_components)
    x_pca = pca.fit_transform(data) 
    print('explained_variance_ratio_', pca.explained_variance_ratio_)
    print('Eigenvalues', pca.explained_variance_)
    #print('Eigenvectors', pca.components_)
    results = pd.DataFrame(pca.components_) 
    results.columns = data.columns
    
    results = results.T
    #cm = sns.light_palette("red", as_cmap=True)
    styler = results.style.background_gradient(cmap='PuBu') # PuBu viridis cm
    #styler.set_precision(3)
    #styler.render()
    if file is not None:
        with open(file, 'w') as f:
            f.write(styler.to_html())
        print('wrote:', file)
    
    return styler

# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=11, window='hanning'):
    #['flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.' + window + '(window_len)')
    return np.convolve(w/w.sum(), s, mode='valid')

def do_pca(data, n_components=2, window='hanning', window_len=14):
    pca = PCA(n_components)
    x_pca = pca.fit_transform(data)
    pca_c = pd.DataFrame(x_pca)     
    vals = x_pca[:,0]
    return smooth(vals, window_len=14, window=window) #7 14 29 57
#    data['pca'] = vals[:605]   
#    data['pca'].plot(kind='line', title=title, grid=True, ax=ax) #driving_distance, total_cycles, total_visits

def get_data(file, dcol='date', to_date=True):
    print(sys._getframe().f_code.co_name, file)
    df = pd.read_csv(file, encoding='utf-8') #latin-1
    #print(df.columns)

    df.drop(df[df[df.columns[-1]] == '-'].index, inplace=True) #"-","2024","9","11","14:00","-"
    df[df.columns[-1]] = df[df.columns[-1]].astype('float64')

    df[dcol] = df['Vuosi'].astype(str) + df['Kuukausi'].astype(str) + df['Päivä'].astype(str) +'-'+ df['Aika [Paikallinen aika]']
    if to_date:
        df[dcol] = pd.to_datetime(df[dcol], format='%Y%m%d-%H:%M')
        df = df.set_index(dcol)
        df = df.sort_values(by=dcol)
    return df

def agg_all(citys, agg):
    df_all = None
    dcol = 'date'
    for city in citys:
        file = os.getcwd() + '/input/saa/{0}_01012024-31122024_{1}.csv'.format(city, agg)
        df = get_data(file, to_date=False)
        vcol = df.columns[-2]
        df = df.rename(columns={vcol:city})[[dcol, city]]
        if df_all is None:
            df_all = df
            continue
        
        df_all = df_all.merge(df, left_on=dcol, right_on=dcol)

    return df_all


# ------------------------------------------------
# main functions

def test_smooth():
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    x = np.array([0,1,1,1,1,1,1,1,1,0])
    x = np.array([0,0,1,1,0,0,1,1,0,0])

    window_len = 5

    xp = pd.DataFrame(x, columns = ['idx'])
    x_hat = xp.rolling(window_len, win_type='gaussian').mean(std=3) # 10
    print(x.shape,'x:', x)

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]] #start:stop:step
    print('window_len:', window_len)
    print(s.shape,'s:', s)

    v = smooth(x, window_len=window_len, window='hanning')
    print(v.shape,'v:', v)
    print('diff:', v.shape[0]-x.shape[0])

    n = (window_len - 1) / 2
    n = math.ceil(n)
    k = 1 if (window_len % 2) == 0 else 0
    v = v[n:-n+k]
    print(v.shape, n, 'v:', v)

    fig = plt.figure(figsize=(10,4))
        
    plt.subplot(131)
    plt.plot(x)
    plt.title("input")
    #plt.xlabel('x')
    #plt.ylabel('y')

    plt.subplot(132)
    plt.plot(v)
    plt.title("hanning")
    ax = fig.add_subplot(133)
    x_hat.plot(title='orig', ax=ax)
    plt.tight_layout()
    #plt.grid()
    plt.show()
    plt.close()

def plot_single_timeseries(file):
    print(sys._getframe().f_code.co_name)
    city = file.split('saa/')[1].split('_')[0] 
    df = get_data(file)
    print(df)
    title = '{0} - {1}'.format(city,df.columns[-1])
    
    df[df.columns[-1]].plot(rot=45, title=title,figsize=(10, 5)) #w,h
    plt.grid()
    plt.show()
    plt.close()
    
def plot_functions(file, city):    
    print(sys._getframe().f_code.co_name)
    df = get_data(file)

    fs = (10, 7)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=fs)#w,h
    ax_flat = axes.flat

    start, end = '2024-01-01', '2024-12-31'

    window = 14
    idx = 0
    vcol = df.columns[-1]

    title = '{0} - {1} (d+rolling)'.format(city, vcol)
    df1 = df.loc[start:end]
    df_group = df1.groupby(pd.Grouper(freq='D'))
    df_mean = df_group[vcol].mean()
    df_mean.plot(kind='line', title=title, ax=ax_flat[idx])          # 1. plot mean of series of daily agg
    ax_flat[idx].grid()
    
    df2 = df.loc[start:end]
    df_group = df2.groupby(pd.Grouper(freq='D'))
    df_mean = df_group[vcol].mean()
    long_rolling = df_mean.rolling(window=window).mean()
    long_rolling.plot(kind='line', ax=ax_flat[idx])                  # 2. plot rolling mean window
    ax_flat[idx].grid()
 
    idx += 1
    title = '{0} - {1} (sum+rolling)'.format(city, vcol)
    df_sum = df_group[vcol].sum()
    #df_sum.plot(kind='line', grid=True, title=title, ax=ax_flat[idx])
    long_rolling = df_sum.rolling(window=window).mean()
    long_rolling.plot(kind='line', ax=ax_flat[idx])                  # 3. plot sum of daily agg
    ax_flat[idx].grid()


    #idx += 1
    periods = 7*2*2 # 4 12 25, 30 90 180
    title = '{0} - {1} (pct_change={2}, window={3})'.format(city, vcol, periods, window)
    pct_change = long_rolling.pct_change(periods=periods)  #  diff pct_change
    pct_change.plot(kind='line', title=title, ax=ax_flat[idx])        # 4. plot percentege of change
    #pct_change = pct_change.rolling(window=window).mean()
    #pct_change.plot(kind='line',  title=title, ax=ax_flat[idx])
    ax_flat[idx].grid()

    idx += 1
    df2 = df.loc[start:end]
    #df2[vcol] = df2[vcol].clip(0, 150) 
    df_group = df2.groupby(pd.Grouper(freq='M'))
    #months = pd.concat([pd.DataFrame(x[1].values) for x in df_group], axis=1)
    months = pd.concat([pd.DataFrame(x[1].values) for x in df_group[vcol]], axis=1)
    months = pd.DataFrame(months)
    months.columns = range(1, months.shape[1]+1)
    #months.columns = range(1,26) 
    months.boxplot(ax=ax_flat[idx])                                              # 5. boxplot per period

    plt.tight_layout()
    #plt.grid()
    plt.show()
    plt.close()

    fs = (10, 9)#w,h
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=fs)
    ax_flat = axes.flat
    idx = 0
    
    window = 2
    df3 = df.loc[start:end]
    df_group = df3.groupby(pd.Grouper(freq='M'))
    
    title = '{0} - {1} (mean window={2})'.format(city, vcol, window)
    df_mean = df_group[vcol].mean()
    df_mean.plot(kind='line',  title=title, ax=ax_flat[idx])       # 6. monthly
    long_rolling = df_mean.rolling(window=window).mean()
    long_rolling.plot(kind='line', ax=ax_flat[idx])                # 7. not useful
    ax_flat[idx].grid()

    #df_mean = df1[vcol]
    df_mean = df3.groupby(pd.Grouper(freq='D'))[vcol]
    idx += 1
    rot = 45
 
    # MACD    
    span_lo, span_hi = 12, 26
    title = '{0} - {1} (MACD {2}/{3})'.format(city, vcol, span_lo, span_hi) #EMA 9
    ema_lo = df_mean.ewm(span=span_lo, adjust=False).mean()        
    ema_hi = df_mean.ewm(span=span_hi, adjust=False).mean() 
    macd = ema_lo - ema_hi
    
    print(macd)
    
    macd.plot(kind='line', grid=True, title=title, ax=ax_flat[idx], rot=rot)    # 8. macd
    ema_9 = macd.ewm(span=9, adjust=False).mean() 
    ema_9.plot(kind='line', ax=ax_flat[idx], rot=rot)                           # 9. ema9
    ax_flat[idx].grid()
    
    
    # RSI
    delta = macd.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down
        
    rsi = 100 - (100/(1 + rs))
    rsi = rsi.iloc[14:]

    idx += 1
    t =  f' (RSI) ' 
    ax = rsi.plot(grid=True, ax=ax_flat[idx], legend=True)                      # 10. rsi
    ax.set_ylim(0,100)
    ax.axhline(70, color='g', linestyle='--', label='overbought')
    ax.axhline(30, color='black', linestyle='--', label='oversold')
    ax.legend()
       
    plt.tight_layout()
    #plt.grid()
    plt.show()
    plt.close()


def test_pca(df, vcols, dcol='date'):
    #df[dcol] = pd.to_datetime(df[dcol], format='%Y%m%d-%H:%M')
    #df = df.set_index(dcol)
    #df.sort_values(by=dcol)  
    #df = df.groupby(pd.Grouper(freq='D'))

    data = df[vcols]
    pca = do_pca(data, n_components=2, window='hanning', window_len=30)

    fs = figsize=(10,7)
    fig = plt.figure(figsize=fs)

    rows,cols = 200,10
    title = 'raw data'
    ax = fig.add_subplot(rows + cols + 1) #nrows, ncols, index
    df[vcols].plot(kind='line', title=title, ax=ax)
    ax.grid()
    
    ax = fig.add_subplot(rows + cols + 2) #nrows, ncols, index
    cpca = 'pca'
    title = cpca
    df[cpca] = pca[0:14469]
    df[cpca].plot(kind='line', title=title, ax=ax)
    ax.grid()

    plt.tight_layout()
    #plt.grid()
    plt.show()
    plt.close()
    
    cols = vcols + [cpca]
    data = df[cols]
    plot_evr(data, n_components=len(cols))
    styler = plot_loadings(df[vcols], n_components=2, file='pca_loadings.html')

def plot_epochs(df, vcols, dcol='date', cpca='pca'):
    from datetime import datetime
    data = df[vcols]
    pca = do_pca(data, n_components=2, window='hanning', window_len=30)
    cpca = 'pca'
    df[cpca] = pca[0:14469]

    df[dcol] = pd.to_datetime(df[dcol], format='%Y%m%d-%H:%M')
    df = df.set_index(dcol)
    df.sort_values(by=dcol) 
    
    idx = df.index.values


    p = '%Y-%m-%d'
    epoch = datetime(1970, 1, 1)
    epochs = []
    for i in idx:
        e = int((datetime.strptime(i, p) - epoch).total_seconds())
        epochs.append(e)

    #pca_vals
    x = epochs
    y = pca_vals
    plt.scatter(x, y, s=10)
    plt.title(f"xxx")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    #plt.grid()
    plt.show()
    plt.close()

if __name__ == '__main__':
    #pd.set_option('display.max_rows', 6000)
    #pd.set_option('display.max_columns', 500)
    #pd.set_option('display.width', 1000)
    
    # https://www.ilmatieteenlaitos.fi/havaintojen-lataus
    # Havaintoasema  Vuosi  Kuukausi  Päivä Aika [Paikallinen aika]  Ilman lämpötila keskiarvo [°C]
    citys = ['helsinki_kaisaniemi', 'jyväskylä_lentoasema', 'oulu_lentoasema', 'vaasa_lentoasema']
    city = citys[0]
    agg = ['daily', 'hourly'][1]
    file = os.getcwd() + '/input/saa/{0}_01012024-31122024_{1}.csv'.format(city, agg)

    #test_smooth()    
    #plot_single_timeseries(file)
    #plot_functions(file, city)
    
    df_all = agg_all(citys, agg)
    test_pca(df_all, citys)
    #plot_epochs(df_all, citys)
    
