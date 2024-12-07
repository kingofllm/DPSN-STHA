import numpy as np

def StandardScaler(train_x, val_x, test_x):
    mean = train_x.mean(axis=(0, 1, 3), keepdims=True)
    std = train_x.std(axis=(0, 1, 3), keepdims=True)
    train_norm = (train_x - mean) / std
    val_norm = (val_x - mean) / std
    test_norm = (test_x - mean) / std
    print(train_norm.shape)
    return train_norm, val_norm, test_norm

def get_sample(data, num_for_predict, num_of_hour, idx):
    hour = None
    if idx + num_for_predict >= data.shape[0]:
        return hour, None
    id = []
    start_idx = idx - num_of_hour
    if start_idx >= 0:
        end_idx = start_idx + num_for_predict
        id.append((start_idx, end_idx))
    else:
        return None, None
    hour = np.concatenate([data[i:j] for i, j in id], axis=0)
    target = data[idx:idx+num_for_predict]
    return hour, target
def load_data(filename, num_for_predict, num_of_hour, batch_size):
    data = np.load(filename)['data']
    all_samples = []
    for idx in range(data.shape[0]):
        sample = get_sample(data, num_for_predict, num_of_hour, idx)
        if sample[0] is None:
            continue
        hour, target = sample
        sample = []
        hour = np.expand_dims(hour, axis=0).transpose((0, 2, 3, 1))
        sample.append(hour)
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))
        sample.append(target)
        id = np.expand_dims(np.array([idx]), axis=0)
        sample.append(id)
        all_samples.append(sample)

    split1 = int(len(all_samples) * 0.6)
    split1 = int(split1 / batch_size) * batch_size
    split2 = int(len(all_samples) * 0.8)
    split2 = int(split2 / batch_size) * batch_size
    
    split3 = int(len(all_samples) / batch_size) * batch_size

    #train_set:[(hour:B,N,C,Th),(target:B,N,Tpre),(id:B,1)]
    train_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split1])]
    val_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split1:split2])]
    test_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split2:split3])]

    train_x = train_set[0]
    val_x = val_set[0]
    test_x = test_set[0]

    train_target = train_set[1]
    val_target = val_set[1]
    test_target = test_set[1]

    "标准化"
    train_x_norm, val_x_norm, test_x_norm = StandardScaler(train_x, val_x, test_x)


    all_data = {
        'train': {
            'train_x': train_x_norm,
            'train_target': train_target,
        },
        'val': {
            'val_x': val_x_norm,
            'val_target': val_target,
        },
        'test': {
            'test_x': test_x_norm,
            'test_target': test_target,
        }
    }
    return all_data



