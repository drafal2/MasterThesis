
import os.path
import pandas as pd
from common.logger import logger
from backtesting.output_analyzer import OutputAnalyzer
from common.my_functions import load_arima_garch_predictions
from os.path import join as pjoin
from common.config import *
import numpy as np
import statsmodels.api as sm
from backtesting.generate_schedules import generate_schedule
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import yfinance as yf
import xgboost


dir_dict = arima_garch_models_dir


class ArimaGarchAnalyzer(OutputAnalyzer):
    """
    Klasa służy analizie wyników prognozy z wykorzystaniem modeli ARIMA, ARIMA-GARCH. Klasa dziedziczy atrybuty
    klasy OutputAnalyzer oraz przy inicjacji wczytuje pliki .csv z przygotowanymi w R prognozami
    """
    def __init__(self, cost, start_capital, start_dte, end_dte, files=None, log_level='INFO', log_to_file=True):
        """
        :param cost: Koszt ponoszony przy otwieraniu pozycji (w %, czyli podając 0.02 koszt to 2%). Jeśli podano
                    wartość 'default', policzony zostanie domyslny koszt z Kryńska and Ślepaczuk (2023)
        :param start_capital: Kapitał z którym zaczynamy inwestycje
        :param start_dte: Data startu prognoz, należy podać w postaci listy
        :param end_dte: Data końca prognoz, należy podać w postaci listy
        :param files: Lista plików do wczytania, jeśli argument nie jest podany to wczytywane są wszystkie pliki ze
                        słownika dir_dict
        :param log_level: Poziom loggera
        :param log_to_file: True / False - czy zapisać logi do pliku
        """

        super().__init__(cost=cost, start_capital=start_capital, start_periods_dates=start_dte, end_periods_dates=end_dte,
                         logger_lvl=log_level, logger_to_file=log_to_file)
        self.class_logger = logger('ArimaGarchAnalyzer', log_level, log_to_file)
        self.class_logger.debug('Zainicjowano klasę')

        self.list_of_files = None

        if files is None:
            self.read_output_from_R()
            self.class_logger.info(f'Wczytano pliki do analizy')
        else:
            self.read_output_from_R(files=files)
            self.class_logger.info(f'Wczytano pliki {files} do analizy')

        self.regimes_df = pd.DataFrame(data={'Regime': [np.NaN]}, index=self.buy_and_hold_df.index)

    def read_output_from_R(self, files=None):
        """
        Metoda wczytuje pliki .csv z prognozami z modeli ARIMA i ARIMA-GARCH
        :param files: Lista plików do wczytania. Jeżeli żaden parametr nie jest podany, to wczytywane są wszystkie
                        pliki ze słownika dir_dict
        """

        self.class_logger.debug('Rozpoczynam pobieranie danych z prognoz modeli ARIMA-GARCH')

        list_of_files = []
        if files is not None:
            for file in files:
                list_of_files.append(dir_dict.get(file))
                path = pjoin(env_path, 'arima_garch_R_output', file)

                if not os.path.exists(path):
                    raise ValueError(f'Podana lokalizacja {path} nie istnieje')

                self.class_logger.debug(f'Plik ze ścieżki {path} zostanie wczytany')

                try:
                    self.__setattr__(dir_dict.get(file),
                                     load_arima_garch_predictions(path, self.class_logger))
                except Exception:
                    self.class_logger.error(f'Plik pod ścieżką {path} nie istnieje')

                self.class_logger.debug(f'Plik {file} został wczytany')
        else:

            for key, value in dir_dict.items():
                list_of_files.append(value)
                self.class_logger.debug(f'Wczytywanie plików: Klucz: {key}, wartość: {value}')

                path = pjoin(env_path, 'arima_garch_R_output/' + key)
                self.class_logger.debug(f'Ścieżka do plików: {path}')

                self.__setattr__(value,
                                 load_arima_garch_predictions(path, self.class_logger))

        self.list_of_files = list_of_files
        self.class_logger.info('Wczytanie outputów z R zakończone sukcesem')

    def predictions_to_signals(self, objs=None, add_to_list=True):
        """
        Metoda generuje sygnały transakcyjne na podstawie prognoz stopy zwrotu
        :param objs: Lista atrybutów z klasy, dla których chcemy wyprodukować sygnały transakcyjne. Jeśli żaden nie
                        podany to produkowane są sygnały dla wszystkich obiektów wczytanych z plików .csv
        """

        if objs is None:
            objs = dir_dict.values()

        for o in objs:
            self.class_logger.debug(f'Dla obiektu {o} będą tworzone sygnały transakcyjne')

            df = getattr(self, o)
            df['Signal'] = df['Predictions'].apply(lambda x: 1 if x > self.cost else (-1 if x < -self.cost else 0))

            # Jeżeli koszt wyższy o prognozowanych zysków, nie zmieniaj pozycji
            ind = df['Signal'] == 0
            sum_0 = sum(ind)

            if sum_0 > 0:
                while sum_0 > 0:
                    prev = df['Signal'].shift(1)
                    df['Signal'].loc[ind] = prev.loc[ind]
                    ind = df['Signal'] == 0
                    sum_0 = sum(ind)

            # Zajmij pozycję w pierwszym dniu niezależnie od kosztu
            if df['Signal'].iloc[0] not in (1, -1):
                df['Signal'].iloc[0] = 1 if df['Predictions'].iloc[1] >= 0 else -1

            self.__setattr__(o, df)
            if add_to_list:
                self.list_of_ready_strategies.append(o)

            self.class_logger.debug(f'Utworzono sygnały transakcyjne dla obiektu {o}')

        self.class_logger.debug('Utworzono sygnały transakcyjne na bazie prognoz ze wskazanych modeli')

    def identify_regimes(self, start_date, end_date, n_regimes=3, primary=True):
        """
        Metoda pozwalająca na identyfikacje reżimów w podanych dniach handlowych
        :param start_date: Data, od której ma być oszacowany model Markova
        :param end_date: Data, do której ma być oszacowany model Markova
        :param n_regimes: Liczba założonych możliwych reżimów zmienności
        :param primary: Czy metoda wykorzystywana jest do przypisania obserwacjom reżimu (True), czy do prognozowania reżimu na kolejny okres (False)
        """

        regimes_window = self.main_df.dropna().copy()
        regimes_window['Returns'] = np.log(regimes_window['ClosePrevious'] / regimes_window['Close'])

        cols = ['Returns']
        regimes_window = regimes_window.loc[(regimes_window.index >= start_date) & (regimes_window.index <= end_date)]
        regimes_window_ret = regimes_window[cols].copy()

        # Próbuj skonwergować model Markova. Jeżeli się nie uda, podnieś błąd. Jego obsługa sprawi, że reżim w kolejnym
        # okresie będzie taki sam jak w obecnym
        converged = False
        max_iter_limit = 1000
        default_max_iter = 200
        accrual = 200
        max_iter = default_max_iter

        model = sm.tsa.MarkovRegression(regimes_window_ret, k_regimes=n_regimes, trend='n', switching_variance=True)

        while not converged and max_iter <= max_iter_limit:
            converged, res_kns = fit_model_with_convergence_handling(model, max_iter)
            if not converged:
                self.class_logger.warning(f"Konwergencja nie osiągnięta przy {max_iter} iteracjach. Zwiększam maksymalną liczbę iteracji o {accrual}.")
                max_iter += accrual

        if converged:
            self.class_logger.debug(f"Model skonwergował przy maksymalnej liczbie iteracji = {max_iter}.")
        else:
            self.class_logger.error(f"Model nie skonwergował przy maksymalnej liczbie iteracji {max_iter_limit}. Prognoza reżimu to reżim z poprzedniego dnia")
            raise ValueError(f'Błąd konwergencji algorytmu optymalizacji modelu Markova! Nie udało się skonwergować przy {max_iter} interacjach.'
                             f'Prognoza reżimu to reżim z poprzedniego dnia')

        # Przypisz reżim, który ma najwyższe prawdopodobieństwo
        params = res_kns.params.loc[[True if 'sigma2' in x else False for x in res_kns.params.index]].reset_index()
        params['index'] = params['index'].str.extract(r'\[(\d+)\]')
        params['index'] = params['index'].astype(float)
        params.columns = ['index', 'variance']
        params['rank'] = params['variance'].rank(method='first', ) - 1
        params['check'] = params['index'] != params['rank']
        regime = res_kns.smoothed_marginal_probabilities.idxmax(axis=1)

        #  We plot the filtered and smoothed probabilities of a recession. Filtered refers to an estimate of the probability at time t
        #  based on data up to and including time t (but excluding time t+1, ... , T
        #  Smoothed refers to an estimate of the probability at time  using all the data in the sample.
        #  https://www.statsmodels.org/devel/examples/notebooks/generated/markov_autoregression.html

        regimes_window['Regime'] = regime
        self.regimes_df = regimes_window

        regimes = regimes_window.Regime.unique()

        if primary:
            self.regimes_dict = dict()

            for r in regimes:
                df = regimes_window.loc[regimes_window.Regime == r]
                new_key = 'regime_' + str(r)
                self.regimes_dict[new_key] = df.index
        else:
            return regimes_window

    def find_best_model_for_regime(self, criterion, n_regimes):  # , train_end=None):
        """
        Metoda szukająca najlepszego modelu prognozującego w danym okresie treningowym
        :param criterion: Kryterium według którego ma być poszukiwany najlepszy model prognozujący
        :param n_regimes: Założona liczba możliwych reżimów
        # :param train_end:
        """

        if len(self.regimes_dict.keys()) != n_regimes:
            raise ValueError(f'W okresie treningowym nie zidentyfikowano {n_regimes} reżimów, tylko {len(self.regimes_dict.keys())} reżimy.'
                             f'Być może wydłużenie okresu treningowego pomoże.')

        # Dla każdego reżimu znajdź najlepszy model prognozujący
        criterion_best = {}
        model_best = {}
        for key, value in self.regimes_dict.items():
            criterion_best[key] = -999
            # if train_end is not None:
            #     value = value[value <= train_end]

            # Ostrzeżenie gdy model został ustalony na podstawie małej liczby obserwacji
            if len(value) < 10:
                self.class_logger.warning(
                    f'Dla {key} zidentyfikowano zaledwie {len(value)} obserwacji. Na tej podstawie zostanie wyznaczony najlepszy model - miej to na uwadzę przy interpretacji')

            # Sprawdź każdy model prognozujący i wybierz najlepszy w danym reżimie wg podanego kryterium
            for f in self.list_of_files:
                model = getattr(self, f)
                df = model.loc[value]

                self.temp_df = df
                self.predictions_to_signals(['temp_df'], add_to_list=False)
                res = self.calculate_investment_value(['temp_df'])

                if criterion_best.get(key) < res.get(criterion):
                    criterion_best[key] = res.get(criterion)
                    model_best[key] = f

        self.criterion_best = criterion_best
        self.model_best = model_best


    def optimise_strategy(self, schedules, interval, moving=True, n_regimes=3):
        """
        Metoda, która przeprowadza backtesting strategii, w której reżim zmienności prognozowany jest naiwnie - reżim w następnym okresie jest taki sam jak w obecnym
        :param schedules: Słownik, w którym skonfigurowano harmonogram okresów treningowych i testowych
        :param interval: Czas trwania okresu testowego przed aktualizacją parametrów i hiperparametrów modeli
        :param moving: Czy okres in-sample ma stałą szerokość 10 lat (True), czy rozszerza się (False)
        :param n_regimes: Liczba założonych możliwych reżimów zmienności
        """

        # Utwórz harmonogramy okresów treningowych i testowych
        train_period_beg_first = schedules.get('train_period_beg_first')
        train_period_beg_last = schedules.get('train_period_beg_last') if moving else schedules.get('train_period_beg_first')
        train_period_end_first = schedules.get('train_period_end_first')
        train_period_end_last = schedules.get('train_period_end_last')
        test_period_beg_first = schedules.get('test_period_beg_first')
        test_period_beg_last = schedules.get('test_period_beg_last')
        test_period_end_first = schedules.get('test_period_end_first')
        test_period_end_last = schedules.get('test_period_end_last')

        interval = interval_dict.get(interval)

        train_period_end = generate_schedule(start_date=train_period_end_first,
                                             end_date=train_period_end_last, interval=interval)
        test_period_start = generate_schedule(start_date=test_period_beg_first,
                                              end_date=test_period_beg_last, interval=interval)
        test_period_end = generate_schedule(start_date=test_period_end_first,
                                            end_date=test_period_end_last, interval=interval)

        if train_period_beg_first == train_period_beg_last:
            train_period_start = [train_period_beg_first] * len(train_period_end)
        else:
            train_period_start = generate_schedule(start_date=train_period_beg_first,
                                                   end_date=train_period_beg_last, interval=interval)


        # Backtesting strategii
        for train_start, train_end, test_start, test_end in zip(train_period_start, train_period_end, test_period_start, test_period_end):
            self.regime_predictions_df = pd.DataFrame({'Regime': np.NaN, 'Best_model': np.NaN, 'Predictions': np.NaN},
                                                      index=self.main_df.index[(self.main_df.index >= test_start) & (self.main_df.index <= test_end)])
            self.class_logger.debug(f'Nastąpi identyfikacja reżimów dla dni z okresu {train_start} - {train_end} w ramach próby treningowej')

            # Zidentyfikuj reżimy i znajdź najlepsze modele predykcyjne
            self.identify_regimes(start_date=train_start, end_date=train_end, n_regimes=n_regimes, primary=True)
            self.find_best_model_for_regime(criterion='AdjIR', n_regimes=n_regimes)

            self.class_logger.info(f'Okres testowy: {train_start} - {train_end}, najlepsze modele per '
                                   f'reżim: {self.model_best}, wartość ich metryki AdjIR: {self.criterion_best}')

            dates_to_predict = self.main_df.index[(self.main_df.index >= test_start) & (self.main_df.index <= test_end)]
            regime_prediction = pd.DataFrame({'Prediction': np.NaN}, index=dates_to_predict)

            regime_prediction.Prediction[0] = self.regimes_df.Regime[-1]
            cnt = 1
            self.class_logger.info(f'Wykonano {cnt}/{len(regime_prediction.index)} prognoz reżimu dla okresu testowego '
                                   f'{test_start} - {test_end}')

            pred = []

            # Dokonaj prognoz reżimów. Naiwna prognoza - reżim w następnym okresie taki sam jak w obecnym
            for date in regime_prediction.index[:-1]:
                try:
                    regimes = self.identify_regimes(start_date=train_start, end_date=date, n_regimes=n_regimes, primary=False)
                    pred.append(regimes.Regime[-1])
                except ValueError:
                    pred.append(pred[-1])

                self.class_logger.info(f'Ostatnia dodana prognoza: {pred[-1]}')

                cnt = cnt + 1
                self.class_logger.info(f'Wykonano {cnt}/{len(regime_prediction.index)} prognoz reżimu dla okresu testowego '
                            f'{test_start} - {test_end}')

            # Przypisz do okresów prognozy
            regime_prediction.Prediction[1:] = pred
            self.regime_predictions_df['Regime'] = regime_prediction.loc[:, 'Prediction']
            self.regime_predictions_df['Regime'] = 'regime_' + self.regime_predictions_df['Regime'].astype(int).astype(str)
            self.regime_predictions_df['Best_model'] = self.regime_predictions_df['Regime'].map(self.model_best)

            if np.NaN in self.regime_predictions_df['Best_model'] or None in self.regime_predictions_df['Best_model']:
                raise ValueError(f'Wśród prognoz znajduje się błędna wartość. Znajduję się w okresie testowym {regime_prediction.index[0]} - {regime_prediction.index[-1]}')

            for regime, model in self.model_best.items():
                idx = self.regime_predictions_df.Regime == regime
                predictions_model = getattr(self, model)
                self.regime_predictions_df.loc[idx, 'Predictions'] = predictions_model.loc[self.regime_predictions_df.index, 'Predictions']

            try:
                df = getattr(self, 'df_final')
                df = pd.concat([df, self.regime_predictions_df['Predictions']])
                self.df_final = df
                self.class_logger.info('Dołączono nowe prognozy do atrybutu self.df_final')
            except AttributeError:
                self.df_final = self.regime_predictions_df['Predictions']
                self.class_logger.info('Utworzono atrybut self.df_final')

            self.class_logger.info(f'W atrybucie self.df_final znajdują się prognozy stopy zwrotu z okresu {self.df_final.index[0]}-{self.df_final.index[-1]}')

    def optimise_xgboost(self, schedules, interval, moving=True, n_regimes=3):
        """
        Metoda, która przeprowadza backtesting strategii, w której reżim zmienności prognozowany jest przez model XGBoost
        :param schedules: Słownik, w którym skonfigurowano harmonogram okresów treningowych i testowych
        :param interval: Czas trwania okresu testowego przed aktualizacją parametrów i hiperparametrów modeli
        :param moving: Czy okres in-sample ma stałą szerokość 10 lat (True), czy rozszerza się (False)
        :param n_regimes: Liczba założonych możliwych reżimów zmienności
        """

        # Utwórz harmonogramy okresów treningowych, walidacyjnych i testowych
        train_period_beg_first = schedules.get('train_period_beg_first')
        train_period_beg_last = schedules.get('train_period_beg_last') if moving else schedules.get('train_period_beg_first')
        train_period_end_first = schedules.get('train_period_end_first')
        train_period_end_last = schedules.get('train_period_end_last')
        valid_period_beg_first = schedules.get('valid_period_beg_first')
        valid_period_beg_last = schedules.get('valid_period_beg_last')
        valid_period_end_first = schedules.get('valid_period_end_first')
        valid_period_end_last = schedules.get('valid_period_end_last')
        test_period_beg_first = schedules.get('test_period_beg_first')
        test_period_beg_last = schedules.get('test_period_beg_last')
        test_period_end_first = schedules.get('test_period_end_first')
        test_period_end_last = schedules.get('test_period_end_last')

        interval = interval_dict.get(interval)

        train_period_end = generate_schedule(start_date=train_period_end_first,
                                             end_date=train_period_end_last, interval=interval)
        test_period_start = generate_schedule(start_date=test_period_beg_first,
                                              end_date=test_period_beg_last, interval=interval)
        test_period_end = generate_schedule(start_date=test_period_end_first,
                                            end_date=test_period_end_last, interval=interval)
        valid_period_start = generate_schedule(start_date=valid_period_beg_first,
                                              end_date=valid_period_beg_last, interval=interval)
        valid_period_end = generate_schedule(start_date=valid_period_end_first,
                                            end_date=valid_period_end_last, interval=interval)

        if train_period_beg_first == train_period_beg_last:
            train_period_start = [train_period_beg_first] * len(train_period_end)
        else:
            train_period_start = generate_schedule(start_date=train_period_beg_first,
                                                   end_date=train_period_beg_last, interval=interval)

        # Policz liczbę iteracji do loggowania
        all_iterations = (len(train_period_start) * len(learning_rate_list) * len(max_depth_list) * len(min_child_weight_list)
                          * len(gamma_list) * len(subsample_list) * len(colsample_bytree_list)
                          * len(reg_alpha_list) * len(reg_lambda_list))
        self.class_logger.info(f'Całkowita liczba iteracji: {all_iterations}')
        cnt_overall = 0

        # Backtesting strategii
        for train_start, train_end, test_start, test_end, valid_start, valid_end in zip(train_period_start, train_period_end, test_period_start, test_period_end, valid_period_start, valid_period_end):

            self.regime_predictions_df = pd.DataFrame({'Regime': np.NaN, 'Best_model': np.NaN, 'Predictions': np.NaN},
                                                      index=self.main_df.index[(self.main_df.index >= test_start) & (self.main_df.index <= test_end)])
            self.class_logger.debug(f'Nastąpi identyfikacja reżimów dla dni z okresu {train_start} - {valid_end} w ramach próby treningowej i walidacyjnej')

            # Zidentyfikuj reżimy i znajdź najlepsze modele predykcyjne
            self.identify_regimes(start_date=train_start, end_date=valid_end, n_regimes=n_regimes, primary=True)
            self.find_best_model_for_regime(criterion='AdjIR', n_regimes=n_regimes)

            best_models_df = pd.DataFrame({'markov_train_start': train_start.isoformat(),
                                           'markov_train_end': valid_end.isoformat(),
                                           'regime': self.model_best.keys(),
                                           'model': self.model_best.values()})

            try:
                df = getattr(self, 'df_regimes_summary')
                df = pd.concat([df, best_models_df])
                self.df_regimes_summary = df
            except AttributeError:
                self.df_regimes_summary = best_models_df


            self.class_logger.info(f'Okres optymalizacji modelu Markova: {train_start} - {valid_end}, najlepsze modele per '
                                   f'reżim: {self.model_best}, wartość ich metryki AdjIR: {self.criterion_best}')

            dates_to_predict = self.main_df.index[(self.main_df.index >= test_start) & (self.main_df.index <= test_end)]
            regime_prediction = pd.DataFrame({'Prediction': np.NaN}, index=dates_to_predict)

            # Pobierz i przygotuj dane wejściowe do XGBoosta
            vix_data = yf.Ticker('^VIX').history(start='1996-01-01', end='2023-12-31')['Close']
            vix_data.index = vix_data.index.tz_localize(None)
            sp500_data = yf.Ticker('^GSPC').history(start='1990-01-01', end='2023-12-31')[['High', 'Low', 'Close', 'Volume']]
            sp500_data.index = sp500_data.index.tz_localize(None)

            sp500_data['high_low_diff'] = np.abs(sp500_data['High'] - sp500_data['Low'])
            sp500_data['close_t-1'] = sp500_data.Close.shift(1)
            sp500_data['log_return_1'] = np.log(sp500_data['Close'] / sp500_data['close_t-1'])
            sp500_data.index = pd.to_datetime(sp500_data.index)
            sp500_data.index = sp500_data.index.tz_localize(None)

            idx = sp500_data.index >= pd.to_datetime('1996-01-01')
            sp500_data = sp500_data.loc[idx, :]

            features_cols = ['log_return_1', 'high_low_diff', 'volume', 'vix_close']
            target_col = ['regime']

            sp500_data = (sp500_data.merge(vix_data, how='left', left_index=True, right_index=True)
                          .rename(columns={'Close_y': 'vix_close', 'Volume': 'volume'}))[features_cols]

            # Znajdź dane w danym okresie treningowym
            idx_train = (sp500_data.index >= train_start) & (sp500_data.index <= train_end)
            idx_valid = (sp500_data.index >= valid_start) & (sp500_data.index <= valid_end)
            idx_test = (sp500_data.index >= test_start) & (sp500_data.index <= test_end)
            idx_markov = (sp500_data.index >= train_start) & (sp500_data.index <= valid_end)

            sp500_data_train = sp500_data.loc[idx_train, :]
            sp500_data_valid = sp500_data.loc[idx_valid, :]
            sp500_data_test = sp500_data.loc[idx_test, :]
            sp500_data_markov = sp500_data.loc[idx_markov, :]

            # Oszacuj model Markova
            converged = False
            max_iter_limit = 1000
            default_max_iter = 200
            accrual = 200
            max_iter = default_max_iter

            model = sm.tsa.MarkovRegression(sp500_data_markov.loc[:, 'log_return_1'], k_regimes=n_regimes, trend='n', switching_variance=True)

            while not converged and max_iter <= max_iter_limit:
                converged, res_kns = fit_model_with_convergence_handling(model, max_iter)
                if not converged:
                    self.class_logger.warning(f"Konwergencja nie osiągnięta przy {max_iter} iteracjach. Zwiększam maksymalną liczbę iteracji o {accrual}.")
                    max_iter += accrual

            # Przypisz obserwacjom reżim
            regime = pd.DataFrame(res_kns.smoothed_marginal_probabilities.idxmax(axis=1), columns=['regime'])
            regime = regime.shift(1)

            sp500_data_train = sp500_data_train.merge(regime, how='left', left_index=True, right_index=True)
            sp500_data_train = sp500_data_train.dropna()
            sp500_data_valid = sp500_data_valid.merge(regime, how='left', left_index=True, right_index=True)

            # Przygotuj dane do XGBoost
            train_X = sp500_data_train.loc[:, features_cols]
            train_Y = sp500_data_train.loc[:, target_col]
            valid_X = sp500_data_valid.loc[:, features_cols]
            valid_Y = sp500_data_valid.loc[:, target_col]
            test_X = sp500_data_test.loc[:, features_cols]

            n_classes = int(train_Y.nunique())

            # Znajdź najlepszy model XGBoost
            best_metric = -999
            cnt_period = 0

            for lr in learning_rate_list:
                for depth in max_depth_list:
                    for weight in min_child_weight_list:
                        for gamma in gamma_list:
                            for subsample in subsample_list:
                                for colsample in colsample_bytree_list:
                                    for reg_alpha in reg_alpha_list:
                                        for reg_lambda in reg_lambda_list:
                                            iters_per_period = all_iterations / len(train_period_start)

                                            params = assign_params(learning_rate=lr, max_depth=depth, min_child_weight=weight,
                                                                   gamma=gamma, subsample=subsample, colsample_bytree=colsample,
                                                                   num_class=n_classes, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
                                            xgb = xgboost.XGBClassifier(**params)
                                            eval_set = [(valid_X, valid_Y)]
                                            xgb.fit(train_X, train_Y, eval_set=eval_set, verbose=False)

                                            evals_result = xgb.evals_result()
                                            best_iteration = xgb.best_iteration
                                            best_score = evals_result['validation_0']['auc'][best_iteration]

                                            df_values = pd.DataFrame({'train_period_start': [train_start],
                                                                      'train_period_end': [train_end],
                                                                      'valid_period_start': [valid_start],
                                                                      'valid_period_end': [valid_end],
                                                                      'learning_rate': [lr],
                                                                      'max_depth': [depth],
                                                                      'min_child_weight': [weight],
                                                                      'gamma': [gamma],
                                                                      'subsample': [subsample],
                                                                      'colsample': [colsample],
                                                                      'reg_alpha': [reg_alpha],
                                                                      'reg_lambda': [reg_lambda],
                                                                      'best_iteration': [best_iteration],
                                                                      'auc': [best_score]})

                                            try:
                                                df = getattr(self, 'df_summary')
                                                df = pd.concat([df, df_values])
                                                self.df_summary = df
                                            except AttributeError:
                                                self.df_summary = df_values


                                            if best_metric < best_score:
                                                best_metric = best_score
                                                self.class_logger.info(f'Hyperparameters tuning: nowy najlepszy model o parametrach learning rate: {lr}, '
                                                                       f'max_depth: {depth}, min_child_weight: {weight}, gamma: {gamma}, subsample: {subsample}, '
                                                                       f'colsample: {colsample}, reg_alpha: {reg_alpha}, reg_lambda: {reg_lambda}. '
                                                                       f'Wartość metryki AUC: {best_score}')
                                                best_ml_model = xgb

                                            cnt_overall += 1
                                            cnt_period += 1

                                            if cnt_period % 200 == 0:
                                                self.class_logger.info(f'Wykonano {int(cnt_period)}/{int(iters_per_period)} iteracji w danym okresie testowym.')

                                            if cnt_overall % 500 == 0:
                                                self.class_logger.info(f'Wykonano {int(cnt_overall)}/{int(all_iterations)} wszystkich iteracji.')

            pred_final = best_ml_model.predict(test_X)
            regime_prediction.Prediction = pred_final
            self.regime_predictions_df['Regime'] = regime_prediction.loc[:, 'Prediction']
            self.regime_predictions_df['Regime'] = 'regime_' + self.regime_predictions_df['Regime'].astype(int).astype(str)
            self.regime_predictions_df['Best_model'] = self.regime_predictions_df['Regime'].map(self.model_best)

            # Przypisz obserwacjom prognozę
            for regime, model in self.model_best.items():
                idx = self.regime_predictions_df.Regime == regime
                predictions_model = getattr(self, model)
                self.regime_predictions_df.loc[idx, 'Predictions'] = predictions_model.loc[self.regime_predictions_df.index, 'Predictions']

            try:
                df = getattr(self, 'df_final')
                df = pd.concat([df, self.regime_predictions_df['Predictions']])
                self.df_final = df
                self.class_logger.info('Dołączono nowe prognozy do atrybutu self.df_final')
            except AttributeError:
                self.df_final = self.regime_predictions_df['Predictions']
                self.class_logger.info('Utworzono atrybut self.df_final')

            # Zapisz wyniki cząstkowe
            _name = 'ml_df_final_' + str(interval) + '_' + str(n_regimes) + '_' + str(2013 if moving else 1996) + '_training'
            _name_summ = 'ml_df_final_regimes_summary_' + str(interval) + '_' + str(n_regimes) + '_' + str(2013 if moving else 1996) + '_training'
            _name_ml = 'ml_df_final_ml_summary_' + str(interval) + '_' + str(n_regimes) + '_' + str(2013 if moving else 1996) + '_training'


            self.df_final.to_csv(env_path + '/backtesting/notebooks/summaries/' + _name + '.csv')
            self.df_regimes_summary.to_csv(env_path + '/backtesting/notebooks/summaries/' + _name_summ + '.csv')
            self.df_summary.to_csv(env_path + '/backtesting/notebooks/summaries/' + _name_ml + '.csv')

            self.class_logger.info(f'W atrybucie self.df_final znajdują się prognozy stopy zwrotu z okresu {self.df_final.index[0]}-{self.df_final.index[-1]}')


def assign_params(booster='gbtree',
                  verbosity=1,
                  validate_parameters=False,
                  n_estimators=1000,  # general parameters
                  learning_rate=0.01,  # [0, 1]
                  gamma=0,  # [0, inf),
                  max_depth=5,  # [0, inf)
                  min_child_weight=1,  # [0, inf)
                  max_delta_step=0,  # [0, inf)
                  subsample=0.8,  # [0, 1]
                  colsample_bytree=0.8,  # [0, 1]
                  colsample_bylevel=1,  # [0, 1]
                  colsample_bynode=1,  # [0, 1]
                  reg_lambda=1,  # [0, inf)
                  reg_alpha=1,  # [0, inf)
                  tree_method='exact',  # ['auto', 'exact', 'approx', 'hist']
                  grow_policy='depthwise',
                  max_leaves=0,
                  max_bin=256,
                  num_parallel_tree=1,  # booster parameters
                  objective='multi:softmax',
                  num_class=3,
                  eval_metric='auc',  # learning task parameters
                  early_stopping_rounds=50  # early stopping
                  ):
    """
    Funkcja przypisuje wartość do argumentów podawanych do XGBoost
    """
    param_list = ['booster', 'verbosity', 'validate_parameters', 'n_estimators', 'learning_rate', 'gamma', 'max_depth',
                  'min_child_weight', 'max_delta_step', 'subsample',
                  'colsample_bytree', 'colsample_bylevel', 'colsample_bynode', 'reg_lambda', 'reg_alpha', 'tree_method',
                  'grow_policy', 'max_leaves',
                  'max_bin', 'num_parallel_tree', 'objective', 'num_class', 'eval_metric', 'early_stopping_rounds']

    params = {}
    for parameter in param_list:
        params[parameter] = eval(parameter)

    return params

def fit_model_with_convergence_handling(model, max_iter):
    """
    Funkcja wykorzystywana do znalezienia modelu Markova, który skonwergował
    :param model: Model Markova do optymalizacji
    :param max_iter: Maksymalna liczba iteracji w procesie optymalizacji modelu
    :return: Czy model skonwergował, Model
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        result = model.fit(maxiter=max_iter, disp=False)
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                return False, result
        return True, result
