import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def logData(data):
    data['US GDP'] = np.log1p(data['US GDP'].pct_change())
    data['JP GDP'] = np.log1p(data['JP GDP'].pct_change())
    data['CN GDP'] = np.log1p(data['CN GDP'].pct_change())
    data['US CPI'] = np.log1p(data['US CPI'].pct_change())
    data['JP CPI'] = np.log1p(data['JP CPI'].pct_change())
    data['CN CPI'] = np.log1p(data['CN CPI'].pct_change())
    data['S&P 500'] = np.log1p(data['S&P 500'] .pct_change())
    data['Nikkei'] = np.log1p(data['Nikkei'] .pct_change())
    data['SSE'] = np.log1p(data['SSE'] .pct_change())
    data = data.dropna()
    return data


def adfTest(series):
    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(series, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

    
def ADFTest(data):
    for (columnName, columnData) in data.iteritems():
        print(f'Results of {columnName} Dickey-Fuller Test:')
        adfTest(columnData)
        print()

    
    
def grangerCausality(data, lag):
    """
    A prerequisite for performing the Granger Causality test
    is that the data need to be stationary 
    i.e it should have a constant mean, constant variance, and no seasonal component. 
    Transform the non-stationary data to stationary data by differencing it, 
    either first-order or second-order differencing. 
    Do not proceed with the Granger causality test 
    if the data is not stationary after second-order differencing.
    
    
    The term “Granger-causes” means that knowing the value of time series x 
    at a certain lag is useful for predicting the value of time series y 
    at a later time period.
    
    If we reject the null hypothesis at low p-values, 
    we can conclude that X can contribute some predictive value: 
    X Granger-causes Y by helping to forecast Y.

    Thus, G-causality does not prove a true cause-and-effect chain.
    
    

    """
    from statsmodels.tsa.stattools import grangercausalitytests
    results = grangercausalitytests(data, maxlag = lag, addconst=True, verbose=True)
    print(data.columns)
    print(f'Granger causality test for lag {lag}')
    print(f'F-test p-value: {results[lag][0]["ssr_ftest"][1]}')
    print(f'Chi-squared p-value: {results[lag][0]["ssr_chi2test"][1]}')
    print()
        
def Causality(data,maxlag):
    grangerCausality(data[['S&P 500', 'US GDP']],maxlag)
    grangerCausality(data[['US GDP', 'S&P 500']],maxlag)
    
    grangerCausality(data[['S&P 500', 'US CPI']],maxlag)
    grangerCausality(data[['US CPI', 'S&P 500']],maxlag)
    
    grangerCausality(data[['Nikkei', 'JP GDP']],maxlag)
    grangerCausality(data[['JP GDP', 'Nikkei']],maxlag)
    
    grangerCausality(data[['Nikkei', 'JP CPI']],maxlag)
    grangerCausality(data[['JP CPI', 'S&P 500']],maxlag)
    
    grangerCausality(data[['SSE', 'CN GDP']],maxlag)
    grangerCausality(data[['CN GDP', 'SSE']],maxlag)
    
    grangerCausality(data[['SSE', 'CN CPI']],maxlag)
    grangerCausality(data[['CN CPI', 'SSE']],maxlag)

    """
    granger_causality(data[['US CPI', 'S&P 500']],3)
    granger_causality(data[['JP CPI', 'Nikkei']],3)
    granger_causality(data[['CN CPI', 'SSE']],3)
    """
   

def ACFTest(data):
    from scipy import  stats
    import statsmodels.api as sm  # 统计相关的库
    m = 10 # 我们检验10个自相关系数
    
    acf,q,p = sm.tsa.acf(data,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
    out = np.c_[range(1,11), acf[1:], q, p]
    output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    output = output.set_index('lag')
    temp = np.array(data)
    fig = plt.figure(figsize=(20,5))
    ax1=fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(temp,ax=ax1)
    fig = plt.title(f'{data.columns[0]} autocorrelation')
    print(output)
    return output

def cointegration_test(df, alpha=0.05): 
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.9':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
    print()

def VAR(data,fit_num):
    from statsmodels.tsa.api import VAR
    from sklearn.metrics import mean_squared_error
    nobs = 3
    data_train, data_test = data[0:-nobs], data[-nobs:]
    
    # Check size
    #print(data_train.shape)  # (119, 8)
    #print(data_test.shape)  # (4, 8)

    
    model = VAR(data_train)
    """
    for i in [1,2,3,4,5,6,7,8,9,10]:
        result = model.fit(i)
        print('Lag Order =', i)
        print('AIC : ', result.aic)
        print('BIC : ', result.bic)
        print('FPE : ', result.fpe)
        print('HQIC: ', result.hqic, '\n')
    
    
    """
    x = model.select_order(maxlags=10)
    print(x.summary())
    
    model_fitted = model.fit(fit_num)
    #print(model_fitted.summary())
    
    # Get the lag order
    lag_order = model_fitted.k_ar
    
    # Input data for forecasting
    forecast_input = data.values[-lag_order:]
    
    fc = model_fitted.forecast(y=forecast_input, steps=nobs)
    df_forecast = pd.DataFrame(fc, index=data.index[-nobs:], columns=data.columns + '_2d')
    
    def invert_transformation(df_train, df_forecast, second_diff=False):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:        
            # Roll back 2nd Diff
            if second_diff:
                df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
            # Roll back 1st Diff
            df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
        return df_fc
    
    from statsmodels.tsa.stattools import acf
    def forecast_accuracy(forecast, actual):
        mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
        me = np.mean(forecast - actual)             # ME
        mae = np.mean(np.abs(forecast - actual))    # MAE
        mpe = np.mean((forecast - actual)/actual)   # MPE
        rmse = np.mean((forecast - actual)**2)**.5  # RMSE
        corr = np.corrcoef(forecast, actual)[0,1]   # corr
        mins = np.amin(np.hstack([forecast[:,None], 
                                  actual[:,None]]), axis=1)
        maxs = np.amax(np.hstack([forecast[:,None], 
                                  actual[:,None]]), axis=1)
        minmax = 1 - np.mean(mins/maxs)             # minmax
        return({'mape':mape, 'me':me, 'mae': mae, 
                'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

    df_results = invert_transformation(data_train, df_forecast, second_diff=True)
    
    
    fig, axes = plt.subplots(nrows=int(len(data.columns)/2), ncols=3, dpi=150, figsize=(24,8))
    for i, (col,ax) in enumerate(zip(data.columns, axes.flatten())):
        df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        data_test[col][-nobs:].plot(legend=True, ax=ax);
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=10)
        ax.set_ylim(-0.75,0.75)
        
        
        print(f'Forecast Accuracy of: {col}')
        accuracy_prod = forecast_accuracy(df_results[col+'_forecast'].values, data_test[col])
        for k, v in accuracy_prod.items():
            print(k, ': ', round(v,4))
    
    plt.tight_layout();
        
    return model_fitted

"""
plt.tight_layout();
# Plot
fig, axes = plt.subplots(nrows=4, ncols=2, dpi=40, figsize=(20,15))
for i, ax in enumerate(axes.flatten()):
    ax.plot(data[data.columns[i]], color='red', linewidth=1)
    # Decorations
    ax.set_title(data.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=10)
"""

"""

plt.figure(figsize=(12,5))

ax1 = data['S&P 500'].plot(color='blue', grid=True, label='S&P 500')
ax2 = data['US CPI'].plot(color='red', grid=True, secondary_y=True, label='US CPI')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()


plt.legend(h1+h2, l1+l2, loc=2)
plt.show()
"""
def grangers_causation_matrix(data, variables, maxlag, test='ssr_chi2test', addconst=True, verbose=True): 
    from statsmodels.tsa.stattools import grangercausalitytests
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, addconst=True, verbose=True)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df



#%%


data = pd.read_excel('Data.xlsx', index_col=0)  

USmatrix = grangers_causation_matrix(data[[ 'US GDP','US CPI', 'S&P 500']],variables = data[[ 'US GDP','US CPI', 'S&P 500']].columns,maxlag = 3)
JPmatrix = grangers_causation_matrix(data[['JP GDP', 'JP CPI', 'Nikkei']],variables = data[['JP GDP', 'JP CPI', 'Nikkei']].columns,maxlag = 3)
CNmatrix = grangers_causation_matrix(data[['CN GDP', 'CN CPI', 'SSE']],variables = data[['CN GDP', 'CN CPI', 'SSE']].columns,maxlag = 3)

#Logarithmic transformation 
data = logData(data)
#data= data.diff(1).dropna()
#ADFTest(data)
#Causality(data,5)

cointegration_test(data[[ 'US GDP','US CPI', 'S&P 500']])
cointegration_test(data[['JP GDP', 'JP CPI', 'Nikkei']])
cointegration_test(data[['CN GDP', 'CN CPI', 'SSE']])
#USACF = ACFTest(data[['S&P 500']])
#JSACF = ACFTest(data[['Nikkei']])
#CNACF = ACFTest(data[['SSE']])


#%%

#%%

US_result  = VAR(data[[ 'US GDP','US CPI', 'S&P 500']],1)
JP_result = VAR(data[['JP GDP', 'JP CPI', 'Nikkei']],1)
CN_result  = VAR(data[['CN GDP', 'CN CPI', 'SSE']],4)

def durbin_watsonTest(result, data):
    from statsmodels.stats.stattools import durbin_watson
    out = durbin_watson(result.resid)
    for col, val in zip(data.columns, out):
        print(col, ':', round(val, 2))
        
        
durbin_watsonTest(US_result,data[[ 'US GDP','US CPI', 'S&P 500']])
durbin_watsonTest(JP_result,data[['JP GDP', 'JP CPI', 'Nikkei']])
durbin_watsonTest(CN_result,data[['CN GDP', 'CN CPI', 'SSE']])


#%%
"checking for serial correlation is to ensure that the model is sufficiently able to explain the variances and patterns in the time series."

irf = US_result.irf(6)
irf.plot(orth=True, signif=0.05)

irf = JP_result.irf(6)
irf.plot(orth=True, signif=0.05)

irf = CN_result.irf(15)
irf.plot(orth=True, signif=0.05)
