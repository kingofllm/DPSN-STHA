import pandas as pd
import holidays
import numpy as np
from interval import Interval

def Time_coraviates(start, end, freq):
    date_information = pd.DataFrame({'datetime': pd.date_range(start=start, end=end, freq=freq)})
    date_information.drop([date_information.shape[0] - 1], inplace=True)
    date_information['hour'] = date_information.datetime.dt.time
    date_information['week'] = 0
    date_information['week1'] = date_information.datetime.dt.dayofweek
    "假期"
    f = lambda x: (x.date() in holidays.US()) * 1
    date_information['holiday'] = date_information.datetime.apply(f)

    date_information['peak_hours'] = 0

    date_information = date_information.astype('str')
    "高峰时段，工作日 7：00-9：00、11：30-13：30和17：00-20：00"
    morning_time_interval = Interval("07:00:00", "09:00:00")
    midday_time_interval = Interval("11:30:00", "13:30:00")
    dinner_time_interval = Interval("17:00:00", "20:00:00")
    for i in date_information[(date_information.week1 == '5') | (date_information.week1 == '6')].index.values:
        date_information.iloc[i, 2] = '1'

    "先选出所有日期的索引值，然后再选取时间段"
    for i in date_information[(date_information.week == '0') | (date_information.holiday == '1')].index.values:
        if date_information.iloc[i, :].hour in morning_time_interval or date_information.iloc[i, :].hour in\
           midday_time_interval or date_information.iloc[i, :].hour in dinner_time_interval:
            date_information.iloc[i, 5] = '1'

    date_information.drop(['datetime'], axis=1, inplace=True)
    date_information.drop(['hour'], axis=1, inplace=True)
    date_information.drop(['week1'], axis=1, inplace=True)
    date_information = date_information.to_numpy()
    date_information = date_information.astype('int')
    "16992,3"
    return date_information

def get_sample(date_information, num_for_predict, num_of_hour, idx):
    hour = None
    if idx + num_for_predict >= date_information.shape[0]:
        return hour, None
    id = []
    start_idx = idx - num_of_hour
    if start_idx >= 0:
        end_idx = start_idx + num_for_predict
        id.append((start_idx, end_idx))
    else:
        return None, None
    hour = np.concatenate([date_information[i:j] for i, j in id], axis=0)
    return hour

def Tcov(start, end, freq, num_for_predict, num_of_hour, batch_size):
    date_information = Time_coraviates(start=start, end=end, freq=freq)
    all_samples = []
    for idx in range(date_information.shape[0]):
        sample = get_sample(date_information, num_for_predict, num_of_hour, idx)
        if sample[0] is None:
            continue
        hour = sample
        sample = []
        hour = np.expand_dims(hour, axis=0)
        sample.append(hour)
        all_samples.append(sample)

    split1 = int(len(all_samples) * 0.6)
    split1 = int(split1 / batch_size) * batch_size
    split2 = int(len(all_samples) * 0.8)
    split2 = int(split2 / batch_size) * batch_size

    tcov_train_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split1])]
    tcov_val_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split1:split2])]
    tcov_test_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split2:])]

    tcov_train_x = tcov_train_set[0]
    tcov_val_x = tcov_val_set[0]
    tcov_test_x = tcov_test_set[0]

    all_tcov_data = {
        'tcov_train': {
            'tcov_train_x':  tcov_train_x,
        },
        'tcov_val': {
            'tcov_val_x': tcov_val_x,
        },
        'tcov_test': {
            'tcov_test_x': tcov_test_x,
        },
    }
    return all_tcov_data












