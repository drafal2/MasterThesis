
import pandas as pd
import pandas_market_calendars as m_cal
from dateutil.relativedelta import relativedelta


def generate_schedule(start_date, end_date, interval=12, exchange='NYSE'):
    """
    Funkcja generująca harmonogramy do backtestingu. Należy pamiętać, aby podawać start_date i end_date
            w tej samej konwencji, czyli jeżeli chcemy zwrócić z funkcji początki okresów, to należy podać
            datę początku miesiąca (start_date.day == 1 & end_date.day == 1),
            jeżeli końce okresów, to należy podać datę końca miesiąca (start_date.day == 31 & end_date.day == 31)
    :param start_date: Pierwsza data harmonogramu
    :param end_date: Ostatnia data harmonogramu
    :param interval: Długość trwania pojedynczego okresu, np. 3 - pojedynczy okres będzie trwał 3 miesiące
    :param exchange: Kalendarz giełdy, który ma być brany pod uwagę przy tworzeniu harmonogramu
    ---
    :return: Element harmonogramu backtestingu
    """

    start_date_original = start_date
    y_diff = relativedelta(end_date, start_date)
    y_diff = y_diff.months + y_diff.years * 12
    y_diff = int(y_diff / interval)

    schedule = []

    if start_date.day == 1:
        for month in range(y_diff):
            date = start_date + relativedelta(months=month * interval)
            schedule.append(date)
    else:
        start_date = start_date + relativedelta(days=1)
        for month in range(y_diff):
            date = start_date + relativedelta(months=month * interval) - relativedelta(days=1)
            schedule.append(date)

    schedule.append(end_date)

    exchange_calendar = m_cal.get_calendar(exchange)
    schedule_to_return = []

    for date in schedule:
        date_d = date.day
        date_m = date.month
        date_y = date.year

        start_date_fun = pd.to_datetime(str(date_y) + '-' + str(date_m) + '-01')
        end_date_fun = pd.to_datetime(str(date_y if date_m != 12 else date_y+1) + '-' + str(date_m + 1 if date_m != 12 else 1) + '-01') + relativedelta(days=-1)

        selected_dates = exchange_calendar.schedule(start_date=start_date_fun,
                                                    end_date=end_date_fun).index
        idx_len = len(selected_dates) - 1
        idx = 0 if start_date_original.day == 1 else idx_len
        schedule_to_return.append(selected_dates[idx])

    return schedule_to_return
