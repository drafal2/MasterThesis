import logging
from datetime import datetime
import os
from common.config import env_path
from os.path import join as pjoin
import shutil


class MyLogger(logging.Logger):
    """
    Klasa implementująca możliwość śledzenia wykonywanego kodu i jego analizę. Metody .debug(), .info(),
    .warning(), .error(), .critical() pozwalają na zarządzanie typami loggów, które nas interesują.
    Metodzie .custom_log() poświęcony jest oddzielny opis.
    """
    def debug(self, message):

        if self.level <= 10:
            self.custom_log(message)

    def info(self, message):

        if self.level <= 20:
            self.custom_log(message)

    def warning(self, message):

        if self.level <= 30:
            self.custom_log(message)

    def error(self, message):

        if self.level <= 40:
            self.custom_log(message)

    def critical(self, message):

        if self.level <= 50:
            self.custom_log(message)

    def custom_log(self, message_from_method):
        """
        Metoda zaimplementowana w celu wydruku pożądanej informacji w logu oraz zapisaniu ich
        w pliku tekstowym jeśli egzekutor kodu tego oczekuje.

        :param message: Parametr w którym określona jest wiadomość do przekazania przez logger
        :return: Nothing
        """

        message = (datetime.now().strftime("%Y-%m-%d %H:%M:%S") + self.logger_const + " " +
                   message_from_method)  # wiadomość do wydrukowania i zapisania w pliku

        if self.log_to_file:  # jeśli zapisujemy logi w pliku
            path = pjoin(env_path, 'logs', self.name + '.txt')

            if os.path.exists(path) and not self.if_current_object_file:  # jeśli istnieje już dany plik to przenosimy go
                current_datetime = datetime.now()                         # do archiwum
                date_time_suffix = current_datetime.strftime("_%H%M")

                path_no_file = os.path.splitext(os.path.dirname(path))[0]

                archive_folder = pjoin(path_no_file, 'archive', current_datetime.strftime('%Y%m%d'))
                if not os.path.exists(archive_folder):
                    os.makedirs(archive_folder)

                file_name, file_extension = os.path.splitext(os.path.basename(path))
                new_file_name = f"{file_name}{date_time_suffix}{file_extension}"
                destination_path = os.path.join(archive_folder, new_file_name)
                shutil.move(path, destination_path)

            with open(path, 'a') as file:  # zapisanie logu w pliku tekstowym
                file.write(message + '\n')
                self.if_current_object_file = True

        self.log(logging.INFO, message)  # drukujemy wiadomość w terminalu, zapisanie do pliku .txt dzieje się wcześniej


def logger(name, level, log_to_file=True):
    """
    Funkcja umożliwiająca raportowaniu elementów wykonanego kodu
    :param name: Nazwa zadania drukowana z loggera
    :param level: Poziom, z którego logi chcemy otrzymywać. Dostępne: DEBUG, INFO, WARNING, ERROR, CRITICAL
    :param log_to_file: True albo False, czy zapisujemy plik .txt z logiem
    :return: Obiekt MyLogger dziedziczący po logging.Logger
    """

    level = level.upper()
    try:
        level_ = getattr(logging, level)
    except AttributeError:
        print('Available logger levels are DEBUG, INFO, WARNING, ERROR, CRITICAL')

    logging.basicConfig(format="%(asctime)s -- %(levelname)s -- %(lineno)d -- %(message)s")

    mlogger = MyLogger(__name__)
    mlogger.setLevel(level_)
    mlogger.log_to_file = log_to_file
    mlogger.name = name
    mlogger.if_current_object_file = False  # po to, żeby zarchiwizować istniejący już plik o takiej samej nazwie (logger)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level_)
    mlogger.addHandler(stream_handler)

    mlogger.logger_const = (' -- ' + str(logging.getLevelName(mlogger.level)) + ' -- ' + name + ': ')

    return mlogger


if __name__ == '__main__':
    my_logger = logger('ClassName', 'info')
    my_logger.info('Udało się?')
