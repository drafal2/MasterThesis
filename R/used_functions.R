


arima.garch.pred <- function(start.date = NULL, end.date = NULL, train.window.len = 1000, max_p = 5, max_q = 5,
                             variance.model.type = 'sGARCH', garch.dist = 'ged'){
  
  # start.date            - data początku okresu testowego dla którego chcemy zrobić prognozę
  # end.date              - data końca okresu testowego dla którego chcemy zrobić prognozę
  # train.window.len      - długość okresu treningowegow dniach
  # max_p                 - maksymalna wartość opóźnienia p
  # max_q                 - maksymalna wartość opóźnienia q
  # variance.model.type   - typ modelu GARCH
  # garch.dist            - rozkład składnika losowego w szacowanym modelu GARCH
  
  
  # Funkcja zwraca data.frame z datami i prognozami na poszczególne dni z modelu ARIMA-GARCH
  
  
  if((is.null(start.date) || is.null(end.date))){
    stop('Podaj początek i koniec okresu prognozy !!!')
  }
  
  pred.start.time <- Sys.time()
  
  sp500.ts = xts(sp500$GSPC.Close, sp500$date)
  sp500.diff.ts = diff.xts(sp500.ts, log=T)[2:length(sp500.ts)]
  
  
  pred.first.day = which(index(sp500.diff.ts) == start.date)
  pred.last.day = which(index(sp500.diff.ts) == end.date)
  
  if(length(pred.first.day) == 0 || length(pred.last.day) == 0){
    stop('Podaj daty dla których jest dostępne kwotowanie !!!')
  }
  
  
  train.window.start = pred.first.day - train.window.len
  train.window.end = pred.first.day - 1
  
  
  pred = data.frame('date' = index(sp500.diff.ts)[(pred.first.day):pred.last.day],
                    'prediction' = NA)
  
  
  for(i in 1:nrow(pred)){
    
    start.time = Sys.time()
    
    ts <- sp500.diff.ts[train.window.start:train.window.end]
    aic = Inf
    
    for(p in 0:max_p){
      
      for(q in 0:max_q){
        
        if(p == 0 && q == 0){
          next
        }
        
        spec = ugarchspec(variance.model = list(model = variance.model.type, garchOrder = c(1,1)),
                          mean.model = list(armaOrder = c(p, q, include.mean=T)),
                          distribution.model = garch.dist)
        fit = tryCatch(
          ugarchfit(spec, ts, solver = 'hybrid'), 
          error=function(e) e, 
          warning=function(w) w
        )
        
        if(!is(fit, "warning") && !is(fit, 'error')){
          if(infocriteria(fit)[1] < aic){
            aic <- infocriteria(fit)[1]
            final_model <- fit
          }
        }
      }
    }
    
    if(aic == Inf){
      pred$prediction[i] <- pred$prediction[i-1]
    } else{
      pred$prediction[i] <- ugarchforecast(final_model, n.ahead = 1)@forecast$seriesFor
    } 
    
    if(i %% 10 || i == nrow(pred)){
      write.csv(pred, paste0("output/", train.window.len, '_', variance.model.type, '_', garch.dist, '_sp500', '.csv'))
    }

    train.window.start = train.window.start + 1
    train.window.end = train.window.end + 1
    
    end.time = Sys.time()
    cat('\nModel:', variance.model.type, ', Rozkład:', garch.dist, ', Długość okna estymacji:', train.window.len,
        '\nPoczątek pętli numer', i, ':', as.character(start.time), 
        '\nKoniec pętli:', as.character(end.time),
        '\nStatus wykonania:', i, '/', length(pred$date),
        '\n\n')
  }
  
  pred.end.time <- Sys.time()
  cat('Początek prognozy:', as.character(pred.start.time) , ',\nKoniec prognozy:', as.character(pred.end.time), '\n')
  return(pred)
}




arima.pred <- function(start.date = NULL, end.date = NULL, train.window.len = 1000, 
                       max_p = 5, max_q = 5){ 
  
  # start.date            - data początku okresu testowego dla którego chcemy zrobić prognozę
  # end.date              - data końca okresu testowego dla którego chcemy zrobić prognozę
  # train.window.len      - długość okresu treningowegow dniach
  # max_p                 - maksymalna wartość opóźnienia p
  # max_q                 - maksymalna wartość opóźnienia q
  
  
  # Funkcja zwraca data.frame z datami i prognozami na poszczególne dni z modelu ARIMA
  
  
  if((is.null(start.date) || is.null(end.date))){
    stop('Podaj początek i koniec okresu prognozy !!!')
  }
  
  pred.start.time <- Sys.time()
  
  sp500.ts = xts(sp500$GSPC.Close, sp500$date)
  sp500.diff.ts = diff.xts(sp500.ts, log=T)[2:length(sp500.ts)]
  
  pred.first.day = which(index(sp500.diff.ts) == start.date)
  pred.last.day = which(index(sp500.diff.ts) == end.date)
  
  if(length(pred.first.day) == 0 || length(pred.last.day) == 0){
    stop('Podaj daty dla których jest dostępne kwotowanie !!!')
  }
  
  train.window.start = pred.first.day - train.window.len
  train.window.end = pred.first.day - 1
  
  pred = data.frame('date' = index(sp500.diff.ts)[(pred.first.day):pred.last.day],
                    'prediction' = NA)
  
  for(i in 1:nrow(pred)){

    start.time = Sys.time()
    
    ts <- sp500.diff.ts[train.window.start:train.window.end]
    aic = Inf
    
    for(p in 0:max_p){
      
      for(q in 0:max_q){
        
        if(p == 0 && q == 0){
          next
        }
        
        fit = tryCatch(
          Arima(ts, order = c(p, 0, q), include.mean = T),
          error=function(e) e, 
          warning=function(w) w
        )
        
        if(!is(fit, "warning") && !is(fit, 'error')){
          if(fit$aic < aic){
            aic <- fit$aic
            final_model <- fit
          }
        }
      }
    }
    
    if(aic == Inf){
      pred$prediction[i] <- pred$prediction[i-1]
    } else{
      pred$prediction[i] <- forecast(final_model, 1)$mean
    }
    
    if(i %% 10 || i == nrow(pred)){
      write.csv(pred, paste0("output/", train.window.len, '_arima', '.csv'))
    }
    
    train.window.start = train.window.start + 1
    train.window.end = train.window.end + 1
    
    end.time = Sys.time()
    cat('\nModel: ARIMA',
        '\nPoczątek pętli numer', i, ':', as.character(start.time), 
        '\nKoniec pętli:', as.character(end.time),
        '\nStatus wykonania:', i, '/', length(pred$date),
        '\n\n')
  }
  
  pred.end.time <- Sys.time()
  cat('Początek prognozy:', as.character(pred.start.time) , ',\nKoniec prognozy:', as.character(pred.end.time), '\n')
  return(pred)
}

