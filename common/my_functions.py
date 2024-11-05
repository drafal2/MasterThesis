
import pandas as pd


def load_arima_garch_predictions(file_path, logger):
    """
    Funkcja wczytuje plik .csv z predykcjami z modeli ARIMA, ARIMA-GARCH z podanej ścieżki
    :param file_path: Ścieżka do pliku do wczytania
    :param logger: logger do którego wysyłane mają być logi z wykonania kodu
    :return: pd.DataFrame z predykcjami
    """

    dtype_mapping = {'Predictions': float}
    input = pd.read_csv(file_path, dtype=dtype_mapping, parse_dates=['Date'])
    if input.shape[1] == 1:
        input = pd.read_csv(file_path, sep=';', dtype=dtype_mapping, parse_dates=['Date'])

    output = input.iloc[:, 1:].set_index(['Date'])

    input_len = len(input)
    logger.debug(f'load_arima_garch_predictions(): Liczba rekordów: {input_len}, start_date: {output.index[0]}, end date: {output.index[-1]}')

    return output


if __name__ == "__main__":

    from os.path import join as pjoin
    from common.config import env_path
    from common.logger import logger

    path = pjoin(env_path, 'arima_garch_R_output', '1000_arima.csv')

    load_arima_garch_predictions(path, logger('ArimaGarchAnalyzerTest', 'DEBUG', False))
