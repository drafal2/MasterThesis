
import pandas as pd
import numpy as np
from common.logger import logger
import matplotlib.pyplot as plt
import yfinance as yf


class OutputAnalyzer:
    """
    Klasa powstała na potrzebę analizy wyników strategii. Przy jej inicjacji powstaje już obiekt benchmarkowy,
    z którym można porównywać wartości za pomocą serii zaimplementowanych metod. Dziedziczące klasę nowe klasy
    mają możliwość porówania wyników z benchmarkiem
    """

    def __init__(self, cost, start_capital, start_periods_dates, end_periods_dates, test_periods_start=None,
                 test_periods_end=None, logger_lvl='INFO', logger_to_file=True):
        """
        :param cost: Koszt ponoszony przy otwieraniu pozycji (czyli podając 0.02 koszt to 2%)
        :param start_capital: Kapitał, z którym zaczynamy inwestycje
        :param start_periods_dates: Lista dat, od których zaczynają się okresy treningowe w backtestingu
        :param end_periods_dates: Lista dat, na których kończą się okresy treningowe w backtestingu
        :param test_periods_start: Lista dat, od których zaczynają się okresy testowe w backtestingu
        :param test_periods_end: Lista dat, na których kończą się okresy testowe w backtestingu
        :param logger_lvl: Poziom loggera
        :param logger_to_file: True / False - czy zapisać logi do pliku
        """

        self.class_logger = logger(name='OutputAnalyzer', level=logger_lvl, log_to_file=logger_to_file)
        self.strategy_output_df = None
        self.cost = cost
        self.start_capital = start_capital
        self.schedule_start = start_periods_dates
        self.schedule_end = end_periods_dates
        self.test_schedule_start = test_periods_start
        self.test_schedule_end = test_periods_end
        self.strategies_metrics = None
        self.list_of_ready_strategies = []

        self.class_logger.debug('Zainicjowano klasę')

        df = yf.Ticker('^GSPC').history(start='1993-01-01', end='2023-12-31')[['Close']]
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df.rename(columns={'GSPC.Close': 'Close'}, inplace=True)
        df['ClosePrevious'] = df['Close'].shift(1)
        df = df[['ClosePrevious', 'Close']]

        self.main_df = df
        self.df_IV = None
        self.class_logger.debug('Wczytano dane wejściowe S&P500 Index')

        self.buy_and_hold_df = self.buy_and_hold()

        self.class_logger.info(f'Wczytano dane do atrybutu buy_and_hold_df, koszt transakcyjny: {self.cost * 100}%')

    def buy_and_hold(self):
        """
        Metoda liczy wynik strategii Buy&Hold oraz jej metryki i dodaje je do DataFrame'u self.strategies_metrics.
                    Przygotowuje również obiekty pod dalszą analizę w innych metodach
        """
        try:
            if self.test_schedule_start is None:
                start_dte = self.schedule_start[0]
                end_dte = self.schedule_end[-1]
            else:
                start_dte = self.test_schedule_start[0]
                end_dte = self.test_schedule_end[-1]
        except TypeError:
            raise TypeError('Datę początkową i końcową analizy należy podać jako list()')

        df = self.main_df.copy()
        df_IV = df[(df.index >= start_dte) & (df.index <= end_dte)]

        self.df_IV = df_IV

        # Ustal koszt transakcyjny
        try:
            if self.cost.lower() == 'default':
                self.cost = (2 + 0.5 * 12.5) / (df_IV['Close'].mean() * 50)
                self.class_logger.debug(f'Koszt dla klasy ustawiony na poziomie domyślnym: {self.cost}')
        except Exception:
            self.class_logger.debug(f'Koszt dla klasy ustawiony na poziomie {self.cost}')

        df = df_IV.copy()

        # Policz wynik strategii benchmarkowej
        start_price = df['Close'].iloc[0]
        for date, price in zip(df.index, df['Close']):
            daily_return = (price - start_price) / start_price

            if date == df.index[0]:  # Jeżeli pierwsza transakcja, ponieś koszt
                starting_capital = self.start_capital * (1 - self.cost)
                starting_capital += starting_capital * daily_return

                self.class_logger.debug(f'Index zerowy wykryty, starting_capital={starting_capital}')

                result_df = pd.DataFrame({'Date': [date], 'InvestmentValue': [starting_capital]})
                continue
            else:
                starting_capital += starting_capital * daily_return
                self.class_logger.debug(f'Data: {date}, Cena: {price}, Starting_capital={starting_capital}')

            start_price = price
            result_df = pd.concat([result_df, pd.DataFrame({'Date': [date], 'InvestmentValue': [starting_capital]})],
                                      ignore_index=True)

        self.class_logger.info(f'Zakończono obliczanie strategii Buy&Hold od {start_dte} do {end_dte}, końcowa wartość '
                               f'inwestycji: {starting_capital}')

        result_df.set_index('Date', drop=True)

        # Policz metryki oceniające performance
        comp_arc = self.arc(df=result_df)
        comp_asd = self.asd(df=result_df)
        comp_max_dd = self.max_dd(df=result_df)

        df_strategies = pd.DataFrame(columns=['ARC', 'ASD', 'MaxDD', 'IR', 'AdjIR'])
        strategies_metrics = {'ARC': comp_arc, 'ASD':  comp_asd, 'MaxDD': comp_max_dd,
                              'IR': self.information_ratio(comp_arc, comp_asd),
                              'AdjIR': self.adj_information_ratio(comp_arc, comp_asd, comp_max_dd)}

        df_strategies = df_strategies._append(pd.Series(strategies_metrics, name='Buy&Hold'))
        self.strategies_metrics = df_strategies

        return result_df.set_index(['Date'], drop=True)

    def arc(self, obj='NoObject', df=None):
        """
        Metoda liczy metrykę Annualised Return Compounded (ARC)
        :param obj: Obiekt zawierający kolumnę InvestmentValue pozwalającą na policzenie ARC
        :param df: Jeżeli nie jest podany argument obj, należy podać pd.DataFrame z kolumną InvestmentValue
        :return: Wartość metryki Annualised Return Compounded
        """

        obj = self.check_correctness_of_data(obj, df)

        computed_arc = (obj['InvestmentValue'].iat[-1] / self.start_capital) ** (252 / len(obj)) - 1
        self.class_logger.debug(f'ARC strategii to {computed_arc}')
        return computed_arc

    def asd(self, obj='NoObject', df=None):
        """
        Metoda liczy metrykę Annualised Standard Deviation (ASD)
        :param obj: Obiekt zawierający kolumnę InvestmentValue pozwalającą na policzenie ASD
        :param df: Jeżeli nie jest podany argument obj, należy podać pd.DataFrame z kolumną InvestmentValue
        :return: Wartość metryki Annualised Standard Deviation
        """

        obj = self.check_correctness_of_data(obj, df)

        if 'StrategyDailyReturn' not in obj.columns:
            obj['StrategyDailyReturn'] = np.log(obj['InvestmentValue'] / obj['InvestmentValue'].shift(1))

        # instrukcja warunkowa na potrzeby policzenia metryk dla strategii z 3 reżimami (gdy dany reżim występuje tylko w jednej obserwacji)
        computed_asd = obj['StrategyDailyReturn'].std(skipna=True, ddof=1 if len(obj) > 1 else 0) * np.sqrt(252)
        self.class_logger.debug(f'ASD strategii to {computed_asd}')
        return computed_asd

    def max_dd(self, obj='NoObject', df=None):
        """
        Metoda liczy metrykę Maximum Drawdown (MaxDD)
        :param obj: Obiekt zawierający kolumnę InvestmentValue pozwalającą na policzenie MaxDD
        :param df: Jeżeli nie jest podany argument obj, należy podać pd.DataFrame z kolumną InvestmentValue
        :return: Wartość metryki Maximum Drawdown
        """

        obj = self.check_correctness_of_data(obj, df)

        max_dd_max = pd.Series([self.start_capital] + obj['InvestmentValue'].tolist())
        max_dd_max = max_dd_max.cummax()

        max_dd_min = pd.Series([self.start_capital] + obj['InvestmentValue'].tolist())
        max_dd_min = max_dd_min[::-1].cummin()[::-1]

        max_dd = (max_dd_max - max_dd_min) / max_dd_max
        computed_max_dd = max_dd.max()
        self.class_logger.debug(f'MaxDD strategii to {computed_max_dd}')

        return computed_max_dd

    def information_ratio(self, arc, asd):
        """
        Metoda liczy metrykę Information Ratio (IR)
        :param arc: Policzona wcześniej metryka ARC
        :param asd: Policzona wcześniej metryka ASD
        :return: Wartość metryki Information Ratio
        """

        computed_ir = arc/asd
        self.class_logger.debug(f'Information Ratio strategii to {computed_ir}')
        return computed_ir

    def adj_information_ratio(self, arc, asd, max_dd):
        """
        Metoda liczy metrykę Adjusted Information Ratio (AdjIR)
        :param arc: Policzona wcześniej metryka ARC
        :param asd: Policzona wcześniej metryka ASD
        :param max_dd: Policzona wcześniej metryka MaxDD
        :return: Wartość metryki Adjusted Information Ratio
        """

        if max_dd == 0:
            return self.information_ratio(arc, asd)

        computed_air = (arc ** 2 * np.sign(arc)) / (asd * max_dd)
        self.class_logger.debug(f'Adjusted Information Ratio strategii to {computed_air}')
        return computed_air

    def calculate_investment_value(self, objs):
        """
        Metoda liczy metryki oceniające strategię oraz wartość inwestycji, przygotowując pd.DataFrame
                    z którego będzie rysowany wykres linii equity
        :param objs: Parametr podający strategie do analizy. Muszą być to atrybuty istniejące w tej klasie
        """

        df = self.df_IV.copy()
        df['DailyReturn'] = (df['Close'] - df['ClosePrevious']) / df['ClosePrevious']

        for obj in objs:

            # Zabezpieczenie na wypadek dodania do listy objs nowego obiektu już po wywołaniu metody po raz pierwszy
            if hasattr(self, obj + '_original'):
                obj_to_get = getattr(self, obj + '_original').copy()
            else:
                obj_to_get = getattr(self, obj).copy()
                setattr(self, obj + '_original', obj_to_get)

            self.class_logger.debug(f'Kalkulacja metryk dla atrybutu {obj}')
            df_merged = pd.merge(df.copy(), obj_to_get, how='inner', left_index=True, right_index=True)

            # Sprawdzenie czy poprawnie złączono DataFrame'y
            if obj != 'temp_df':
                if len(df) != len(df_merged):
                    print(df.copy().index)
                    print(obj_to_get.index)
                    raise ValueError(f'Łącząne obiekty pd.DataFrame nie mają wszystkich indeksów (dat) wspólnych, obiekt: {obj}')
                self.class_logger.debug(f'InvestmentValue: Tabele z prognozą {obj} i bazowa df_IV złączone poprawnie')

            df_merged['StrategyDailyReturn'] = df_merged['DailyReturn'] * df_merged['Signal']

            inv_value = self.start_capital
            signal_prev = 0
            inv_list = []
            cost_lst = []
            cost_cum = 0
            cost_cum_lst = []

            # Policz wyniki strategii
            for date in df_merged.index:
                signal = df_merged.at[date, 'Signal']
                s_daily_return = df_merged.at[date, 'StrategyDailyReturn']
                base = inv_value if signal_prev == signal else inv_value * (1 - self.cost)
                cost = 0 if signal_prev == signal else inv_value * self.cost
                cost_lst.append(cost)
                cost_cum += cost
                cost_cum_lst.append(cost_cum)
                inv_value = base * (1 + s_daily_return)
                inv_list.append(inv_value)
                signal_prev = signal
                self.class_logger.debug(f'{obj}: Sygnał transakcyjny: {signal}, dzienny zwrot indeksu: {s_daily_return}, '
                                        f'kapitał za który otwarto pozycję {base}, wynik dzienny: {s_daily_return * base}, '
                                        f'poniesiony koszt: {cost}, koszt skumulowany: {cost_cum}, '
                                        f'wartość inwestycji na dzień {date}: {inv_value}')

            df_merged['InvestmentValue'] = inv_list
            df_merged['Cost'] = cost_lst
            df_merged['CostCumulated'] = cost_cum_lst
            setattr(self, obj, df_merged)

            self.class_logger.debug(f'Atrybut {obj} podmieniony przez pd.DataFrame z kalkulacji InvestmentValue. '
                                    f'Można rysować wykresy')

            # Policz metryki oceniające performance
            comp_arc = self.arc(obj)
            comp_asd = self.asd(obj)
            comp_max_dd = self.max_dd(obj)

            strategies_metrics = {'ARC': comp_arc, 'ASD': comp_asd, 'MaxDD': comp_max_dd,
                                  'IR': self.information_ratio(comp_arc, comp_asd),
                                  'AdjIR': self.adj_information_ratio(comp_arc, comp_asd, comp_max_dd)}

            self.class_logger.debug(f'Zakończono kalkulacje metryk dla {obj}. Końcowa wartość inwestycji: {inv_value}, '
                                   f'metryki: {strategies_metrics}')

            if obj != 'temp_df':
                strategies_metrics = self.strategies_metrics._append(pd.Series(strategies_metrics, name=obj))
                self.strategies_metrics = strategies_metrics

            else:
                del self.temp_df_original
                return strategies_metrics

    def check_correctness_of_data(self, obj, df):
        """
        Metoda sprawdza czy podany obiekt obj isnieje jako atrybut klasy, jeśli nie to sprawdza w podanym pd.DataFrame
                znajduje się kolumna 'InvestmentValue'
        :param obj: Atrybut którego istnienie chcemy sprawdzić
        :param df: pd.DataFrame do weryfikacji jeśli podany obj nie istnieje
        :return: Obiekt nadający się do dalszej analizy
        """
        try:
            obj = getattr(self, obj).copy()
        except AttributeError:
            self.class_logger.debug(
                f'check_correctness_of_data: Podany obiekt {obj} nie isnieje. Próbuje znaleźć obiekt klasy pd.DataFrame, który '
                f'można podać jako argument pod nazwą "df"')
            if df is not None:
                obj = df.copy()
            else:
                self.class_logger.error('check_correctness_of_data: Nie dostarczono obiektu klasy pd.DataFrame')
                raise ValueError(
                    f'Podany atrybut "{obj}" nie jest zdefiniowany. Nie podano również obiektu klasy pd.DataFrame. '
                    f'Obliczenia ARC nie zostanie wykonane')

        if 'InvestmentValue' not in obj.columns:
            self.class_logger.error(
                'check_correctness_of_data: Nie dostarczono odpowiedniego obiektu do obliczeń. '
                'Obiekt musi posiadać kolumnę "InvestmentValue"')
            raise ValueError('pd.DataFrame musi posiadać kolumnę "InvestmentValue"')

        return obj

    def plot_buy_and_hold(self):
        """
        Metoda rysuje wykres wartości inwestycji w czasie w przypadku podejścia Buy&Hold
        """
        df_to_plot = self.buy_and_hold_df.copy()
        df_to_plot.plot()

        self.class_logger.debug('Przygotowano wykres do narysowania')
        plt.show()

    def plot_strategies(self, objects, benchmark=True):
        """
        Metoda rysuje wykres z testowanymi strategiami
        :param objects: Lista atrybutów do narysowania
        :param benchmark: Czy rysować wykres strategii Buy&Hold
        """

        self.class_logger.info(f'Rozpoczynam rysowanie wykresów z atrybutów {objects}')
        labels = []
        objs_to_plot = []

        if benchmark is True:
            self.class_logger.debug('Benchmark Buy&Hold będzie zawarty na wykresie')
            objs_to_plot.append(self.buy_and_hold_df.copy())
            labels.append('Buy&Hold')

        for obj in objects:
            try:
                df = getattr(self, obj)['InvestmentValue'].copy()
            except AttributeError:
                self.class_logger.error(f'Atrybut {obj} nie istnieje w obiekcie, zatem zostanie pominięty')
                continue

            objs_to_plot.append(df)
            labels.append(obj)

        dfs_to_plot = pd.concat(objs_to_plot, axis=1)

        self.dfs_to_plot = dfs_to_plot
        self.class_logger.info('pd.DataFrame na bazie którego narysowano wykres został zapisany jako '
                               'atrybut o nazwie dfs_to_plot i jest łatwo dostępny z poziomu obiektu')


        plt.figure(figsize=(10, 6))
        plt.plot(dfs_to_plot)
        plt.xlabel('Year')
        plt.ylabel('Investment Value')
        plt.gca().set_facecolor('white')
        plt.legend(labels)
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

        ax = plt.gca()
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')

        plt.show()


if __name__ == '__main__':
    x = OutputAnalyzer(0.02, 10000, [pd.to_datetime('2007-01-01')], [pd.to_datetime('2023-06-30')])
    x.plot_buy_and_hold()

