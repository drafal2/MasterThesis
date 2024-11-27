
library(stringr)
library(xts)
library(caTools)
library(e1071)
library(quantmod)
library(lattice)
library(timeSeries)
library(rugarch)
library(forecast)
library(stringr)
library(data.table)
library(tidyquant)
source('used_functions.R')




###############################################################################################################################
##                                                LOADING THE DATA                                                           ## 
###############################################################################################################################



sp500 = getSymbols("^GSPC", from='1990-01-01', to='2023-12-31', auto.assign = FALSE)
x = index(sp500)
sp500 = data.frame(close = sp500$GSPC.Close, date = as.Date(x))

sp500.ts = xts(sp500$GSPC.Close, sp500$date)
sp500.diff.ts = diff.xts(sp500.ts, log=T)[2:length(sp500.ts)]

plot(sp500.diff.ts)
save(sp500, sp500.ts, sp500.diff.ts, file = "data/sp500_files.RData")




###############################################################################################################################
##                                             RUN ARIMA GARCH FORECASTS                                                     ## 
###############################################################################################################################



estimation.windows = c(250, 500, 1000, 1500)
models = c('eGARCH', 'sGARCH')
distibutions = c('snorm', 'sstd', 'ged', 'jsu')


for(w in estimation.windows){
  arima.pred(start.date = as.Date('1996-01-02'), end.date = as.Date('2023-12-29'), train.window.len = w, max_p = 5, max_q = 5)
  for(m in models){
    for(d in distibutions){
      arima.garch.pred(start.date = as.Date('1996-01-02'), end.date = as.Date('2023-12-29'), train.window.len = w, max_p = 5, 
                       max_q = 5, variance.model.type = m, garch.dist =  d)
    }
  }
}

