
import pandas as pd

env_path = 'C:\OneDrive\Pulpit\MasterThesis'

# Przestrzeń hiperparametrów do sprawdzenia
learning_rate_list = [0.01, 0.05, 0.1, 1]
max_depth_list = [4, 7, 10]
min_child_weight_list = [1, 3, 5]
gamma_list = [0, 0.001, 0.1, 1]
subsample_list = [0.5, 0.7, 0.9]
colsample_bytree_list = [0.5, 0.8]
reg_alpha_list = [0.1, 0.5, 1, 10]
reg_lambda_list = [0.1, 0.5, 1, 10]

# Słownik tłumaczący okres na liczbę miesięcy
interval_dict = {'1M': 1,
                 '2M': 2,
                 '3M': 3,
                 '6M': 6,
                 '12M': 12}

# Słownik tłumaczący nazwy plików na atrybuty dla wyników strategii
final_dir = {'df_final_3_1996_2.csv': 'strategy_3_exp_2_reg',
             'df_final_3_2013_2.csv': 'strategy_3_mov_2_reg',
             'df_final_6_1996_2.csv': 'strategy_6_exp_2_reg',
             'df_final_6_2013_2.csv': 'strategy_6_mov_2_reg',
             'df_final_12_1996_2.csv': 'strategy_12_exp_2_reg',
             'df_final_12_2013_2.csv': 'strategy_12_mov_2_reg',
             'ml_df_final_3_1996_2_reg.csv': 'ml_strategy_3_exp_2_reg',
             'ml_df_final_3_2013_2_reg.csv': 'ml_strategy_3_mov_2_reg',
             'ml_df_final_6_1996_2_reg.csv': 'ml_strategy_6_exp_2_reg',
             'ml_df_final_6_2013_2_reg.csv': 'ml_strategy_6_mov_2_reg',
             'ml_df_final_12_1996_2_reg.csv': 'ml_strategy_12_exp_2_reg',
             'ml_df_final_12_2013_2_reg.csv': 'ml_strategy_12_mov_2_reg',
             'ml_df_final_3_2013_3_reg.csv': 'ml_strategy_3_mov_3_reg',
             'ml_df_final_6_2013_3_reg.csv': 'ml_strategy_6_mov_3_reg',
             'ml_df_final_12_2013_3_reg.csv': 'ml_strategy_12_mov_3_reg'
             }

# Słownik tłumaczący nazwy plików z prognozami na atrybuty z prognozami
arima_garch_models_dir = {'250_arima.csv': 'arima_250',
                          '250_eGARCH_ged.csv': 'eGARCH_ged_250',
                          '250_eGARCH_sstd.csv': 'eGARCH_sstd_250',
                          '250_eGARCH_snorm.csv': 'eGARCH_snorm_250',
                          '250_eGARCH_jsu.csv': 'eGARCH_jsu_250',
                          '250_sGARCH_ged.csv': 'sGARCH_ged_250',
                          '250_sGARCH_sstd.csv': 'sGARCH_sstd_250',
                          '250_sGARCH_snorm.csv': 'sGARCH_snorm_250',
                          '250_sGARCH_jsu.csv': 'sGARCH_jsu_250',
                          '500_arima.csv': 'arima_500',
                          '500_eGARCH_ged.csv': 'eGARCH_ged_500',
                          '500_eGARCH_sstd.csv': 'eGARCH_sstd_500',
                          '500_eGARCH_snorm.csv': 'eGARCH_snorm_500',
                          '500_eGARCH_jsu.csv': 'eGARCH_jsu_500',
                          '500_sGARCH_ged.csv': 'sGARCH_ged_500',
                          '500_sGARCH_sstd.csv': 'sGARCH_sstd_500',
                          '500_sGARCH_snorm.csv': 'sGARCH_snorm_500',
                          '500_sGARCH_jsu.csv': 'sGARCH_jsu_500',
                          '1000_arima.csv': 'arima_1000',
                          '1000_eGARCH_ged.csv': 'eGARCH_ged_1000',
                          '1000_eGARCH_sstd.csv': 'eGARCH_sstd_1000',
                          '1000_eGARCH_snorm.csv': 'eGARCH_snorm_1000',
                          '1000_eGARCH_jsu.csv': 'eGARCH_jsu_1000',
                          '1000_sGARCH_ged.csv': 'sGARCH_ged_1000',
                          '1000_sGARCH_sstd.csv': 'sGARCH_sstd_1000',
                          '1000_sGARCH_snorm.csv': 'sGARCH_snorm_1000',
                          '1000_sGARCH_jsu.csv': 'sGARCH_jsu_1000',
                          '1500_arima.csv': 'arima_1500',
                          '1500_eGARCH_ged.csv': 'eGARCH_ged_1500',
                          '1500_eGARCH_sstd.csv': 'eGARCH_sstd_1500',
                          '1500_eGARCH_snorm.csv': 'eGARCH_snorm_1500',
                          '1500_eGARCH_jsu.csv': 'eGARCH_jsu_1500',
                          '1500_sGARCH_ged.csv': 'sGARCH_ged_1500',
                          '1500_sGARCH_sstd.csv': 'sGARCH_sstd_1500',
                          '1500_sGARCH_snorm.csv': 'sGARCH_snorm_1500',
                          '1500_sGARCH_jsu.csv': 'sGARCH_jsu_1500'
                          }

# Słowniki z datami granicznymi okresów testowych, walidacyjnych i treningowych dla strategii ML
ml_dates_1M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
               'train_period_end_first': pd.to_datetime('2002-12-31'),
               'valid_period_beg_first': pd.to_datetime('2003-01-01'),
               'valid_period_end_first': pd.to_datetime('2005-12-31'),
               'test_period_beg_first': pd.to_datetime('2006-01-01'),
               'test_period_end_first': pd.to_datetime('2006-01-31'),
               'train_period_beg_last': pd.to_datetime('2013-12-01'),
               'train_period_end_last': pd.to_datetime('2020-11-30'),
               'valid_period_beg_last': pd.to_datetime('2020-12-01'),
               'valid_period_end_last': pd.to_datetime('2023-11-30'),
               'test_period_beg_last': pd.to_datetime('2023-12-01'),
               'test_period_end_last': pd.to_datetime('2023-12-31')}

ml_dates_2M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
               'train_period_end_first': pd.to_datetime('2002-12-31'),
               'valid_period_beg_first': pd.to_datetime('2003-01-01'),
               'valid_period_end_first': pd.to_datetime('2005-12-31'),
               'test_period_beg_first': pd.to_datetime('2006-01-01'),
               'test_period_end_first': pd.to_datetime('2006-02-28'),
               'train_period_beg_last': pd.to_datetime('2013-11-01'),
               'train_period_end_last': pd.to_datetime('2020-10-31'),
               'valid_period_beg_last': pd.to_datetime('2020-11-01'),
               'valid_period_end_last': pd.to_datetime('2023-10-31'),
               'test_period_beg_last': pd.to_datetime('2023-11-01'),
               'test_period_end_last': pd.to_datetime('2023-12-31')}

ml_dates_3M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
               'train_period_end_first': pd.to_datetime('2002-12-31'),
               'valid_period_beg_first': pd.to_datetime('2003-01-01'),
               'valid_period_end_first': pd.to_datetime('2005-12-31'),
               'test_period_beg_first': pd.to_datetime('2006-01-01'),
               'test_period_end_first': pd.to_datetime('2006-03-31'),
               'train_period_beg_last': pd.to_datetime('2013-10-01'),
               'train_period_end_last': pd.to_datetime('2020-09-30'),
               'valid_period_beg_last': pd.to_datetime('2020-10-01'),
               'valid_period_end_last': pd.to_datetime('2023-09-30'),
               'test_period_beg_last': pd.to_datetime('2023-10-01'),
               'test_period_end_last': pd.to_datetime('2023-12-31')}

ml_dates_6M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
               'train_period_end_first': pd.to_datetime('2002-12-31'),
               'valid_period_beg_first': pd.to_datetime('2003-01-01'),
               'valid_period_end_first': pd.to_datetime('2005-12-31'),
               'test_period_beg_first': pd.to_datetime('2006-01-01'),
               'test_period_end_first': pd.to_datetime('2006-06-30'),
               'train_period_beg_last': pd.to_datetime('2013-07-01'),
               'train_period_end_last': pd.to_datetime('2020-06-30'),
               'valid_period_beg_last': pd.to_datetime('2020-07-01'),
               'valid_period_end_last': pd.to_datetime('2023-06-30'),
               'test_period_beg_last': pd.to_datetime('2023-07-01'),
               'test_period_end_last': pd.to_datetime('2023-12-31')}

ml_dates_12M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
                'train_period_end_first': pd.to_datetime('2002-12-31'),
                'valid_period_beg_first': pd.to_datetime('2003-01-01'),
                'valid_period_end_first': pd.to_datetime('2005-12-31'),
                'test_period_beg_first': pd.to_datetime('2006-01-01'),
                'test_period_end_first': pd.to_datetime('2006-12-31'),
                'train_period_beg_last': pd.to_datetime('2013-01-01'),
                'train_period_end_last': pd.to_datetime('2019-12-31'),
                'valid_period_beg_last': pd.to_datetime('2020-01-01'),
                'valid_period_end_last': pd.to_datetime('2022-12-31'),
                'test_period_beg_last': pd.to_datetime('2023-01-01'),
                'test_period_end_last': pd.to_datetime('2023-12-31')}


ml_schedules_generation_dates = {'1M': ml_dates_1M,
                                 '2M': ml_dates_2M,
                                 '3M': ml_dates_3M,
                                 '6M': ml_dates_6M,
                                 '12M': ml_dates_12M}

# Słowniki z datami granicznymi okresów testowych, walidacyjnych i treningowych dla strategii bez ML
dates_1M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
            'train_period_end_first': pd.to_datetime('2005-12-31'),
            'test_period_beg_first': pd.to_datetime('2006-01-01'),
            'test_period_end_first': pd.to_datetime('2006-01-31'),
            'train_period_beg_last': pd.to_datetime('2013-12-01'),
            'train_period_end_last': pd.to_datetime('2023-11-30'),
            'test_period_beg_last': pd.to_datetime('2023-12-01'),
            'test_period_end_last': pd.to_datetime('2023-12-31')}

dates_2M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
            'train_period_end_first': pd.to_datetime('2005-12-31'),
            'test_period_beg_first': pd.to_datetime('2006-01-01'),
            'test_period_end_first': pd.to_datetime('2006-02-28'),
            'train_period_beg_last': pd.to_datetime('2013-11-01'),
            'train_period_end_last': pd.to_datetime('2023-10-31'),
            'test_period_beg_last': pd.to_datetime('2023-11-01'),
            'test_period_end_last': pd.to_datetime('2023-12-31')}

dates_3M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
            'train_period_end_first': pd.to_datetime('2005-12-31'),
            'test_period_beg_first': pd.to_datetime('2006-01-01'),
            'test_period_end_first': pd.to_datetime('2006-03-31'),
            'train_period_beg_last': pd.to_datetime('2013-10-01'),
            'train_period_end_last': pd.to_datetime('2023-09-30'),
            'test_period_beg_last': pd.to_datetime('2023-10-01'),
            'test_period_end_last': pd.to_datetime('2023-12-31')}

dates_6M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
            'train_period_end_first': pd.to_datetime('2005-12-31'),
            'test_period_beg_first': pd.to_datetime('2006-01-01'),
            'test_period_end_first': pd.to_datetime('2006-06-30'),
            'train_period_beg_last': pd.to_datetime('2013-07-01'),
            'train_period_end_last': pd.to_datetime('2023-06-30'),
            'test_period_beg_last': pd.to_datetime('2023-07-01'),
            'test_period_end_last': pd.to_datetime('2023-12-31')}

dates_12M = {'train_period_beg_first': pd.to_datetime('1996-01-01'),
            'train_period_end_first': pd.to_datetime('2005-12-31'),
            'test_period_beg_first': pd.to_datetime('2006-01-01'),
            'test_period_end_first': pd.to_datetime('2006-12-31'),
            'train_period_beg_last': pd.to_datetime('2013-01-01'),
            'train_period_end_last': pd.to_datetime('2022-12-31'),
            'test_period_beg_last': pd.to_datetime('2023-01-01'),
            'test_period_end_last': pd.to_datetime('2023-12-31')}


schedules_generation_dates = {'1M': dates_1M,
                              '2M': dates_2M,
                              '3M': dates_3M,
                              '6M': dates_6M,
                              '12M': dates_12M}
