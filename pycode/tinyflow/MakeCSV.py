import os
import pandas as pd
import numpy as np

file_list = ['VGG x1', 'VGG x2', 'VGG x3', 'VGG x4','VGG x3','VGG MDW', 'VGG bs4', 'VGG bs8', 'VGG bs32', 'VGG bs64'
             'Inception V3 x1', 'Inception V3 x2', 'Inception V3 x3', 'Inception V3 x4','Inception V3 MDW',
             'Inception V3 bs4', 'Inception V3 bs8', 'Inception V3 bs32', 'Inception V3 bs64',
             'Inception V4 x1', 'Inception V4 x2', 'Inception V4 x3', 'Inception V4 x4', 'Inception V4 MDW',
             'Inception V4 bs4', 'Inception V4 bs8', 'Inception V4 bs32', 'Inception V4 bs64',
             'ResNet x1', 'ResNet x2', 'ResNet x3', 'ResNet x4', 'ResNet MDW', 'ResNet bs4', 'ResNet bs8', 'ResNet bs32', 'ResNet bs64',
             'DenseNet x1', 'DenseNet x2', 'DenseNet x3', 'DenseNet x4', 'DenseNet MDW', 'DenseNet bs4', 'DenseNet bs8', 'DenseNet bs32', 'DenseNet bs64']
single_workloads = ['VGG', 'Inception V3', 'Inception V4', 'ResNet', 'DenseNet']
multi_workloads = ['VGG x1', 'VGG x2', 'VGG x3', 'VGG x4','VGG MDW',
                   'Inception V3 x1', 'Inception V3 x2', 'Inception V3 x3', 'Inception V3 x4','Inception V3 MDW',
                   'Inception V4 x1', 'Inception V4 x2', 'Inception V4 x3', 'Inception V4 x4','Inception V4 MDW',
                   'ResNet x1', 'ResNet x2', 'ResNet x3', 'ResNet x4', 'ResNet MDW',
                   'DenseNet x1', 'DenseNet x2', 'DenseNet x3', 'DenseNet x4','DenseNet MDW']
TENSILE_path = '/home/zkx/PyProject/TENSILE8/pycode/tinyflow/log/'
# todo: MDW
baseline_multi_workloads = ['VGG x1', 'VGG x2', 'VGG x3', 'VGG x4',
                            'InceptionV3 x1', 'InceptionV3 x2', 'InceptionV3 x3', 'InceptionV3 x4',
                            'InceptionV4 x1', 'InceptionV4 x2', 'InceptionV4 x3', 'InceptionV4 x4',
                            'ResNet x1', 'ResNet x2', 'ResNet x3', 'ResNet x4',
                            'DenseNet x1', 'DenseNet x2', 'DenseNet x3', 'DenseNet x4']
baseline_path = '/home/zkx/PyProject/tinyflow-baseline8/tests/Experiment/log/'
batch_size_workloads = ['VGG bs4', 'VGG bs8', 'VGG x1','VGG bs32', 'VGG bs64',
                        'Inception V3 bs4', 'Inception V3 bs8', 'Inception V3 x1', 'Inception V3 bs32', 'Inception V3 bs64',
                        'Inception V4 bs4', 'Inception V4 bs8', 'Inception V4 x1',  'Inception V4 bs32', 'Inception V4 bs64',
                        'ResNet bs4', 'ResNet bs8', 'ResNet x1', 'ResNet bs32', 'ResNet bs64',
                        'DenseNet bs4', 'DenseNet bs8', 'DenseNet x1', 'DenseNet bs32', 'DenseNet bs64']
batch_size_workloads_col = {'VGG x1': 0, 'VGG bs4': 1, 'VGG bs8': 2, 'VGG': 3, 'VGG bs32': 4,
                            'Inception V3 x1': 0, 'Inception V3 bs4': 1, 'Inception V3 bs8': 2, 'Inception V3': 3,
                            'Inception V3 bs32': 4,
                            'Inception V4 x1': 0, 'Inception V4 bs4': 1, 'Inception V4 bs8': 2, 'Inception V4': 3,
                            'ResNet x1': 0, 'ResNet bs4': 1, 'ResNet bs8': 2, 'ResNet': 3, 'ResNet bs32': 4,
                            'DenseNet x1': 0, 'DenseNet bs4': 1, 'DenseNet bs8': 2, 'DenseNet': 3}
title = ['saved_ratio', 'extra_overhead', 'vanilla_max_memory_used', 'schedule_max_memory_used', 'vanilla_time_cost',
         'schedule_time_cost', 'efficiency', '', '',
         'saved_ratio_cold_start', 'extra_overhead_cold_start', 'schedule_max_memory_used_cold_start',
         'efficiency_cold_start']
baseline_title = ['vDNN', 'vdnn_vanilla', 'max_memory', 'time', '', 'max_memory', 'time', 'memory_saved', 'extra_overhead',
                  'efficiency', '', 'capuchin','capu_vanilla', 'max_memory', 'time', '','max_memory', 'time', 'memory_saved', 'extra_overhead',
                  'efficiency']
result_csv_log = '/home/zkx/PyProject/TENSILE8/pycode/tinyflow/log/'


def get_row(path_):
    if 'VGG' in path_:
        row_ = 0
    elif 'InceptionV3' in path_ or 'Inception V3' in path_:
        row_ = 1
    elif 'InceptionV4' in path_ or 'Inception V4' in path_:
        row_ = 2
    elif 'ResNet' in path_:
        row_ = 3
    elif 'DenseNet' in path_:
        row_ = 4
    else:
        raise Exception(f'not supported workload:{path_}')
    return row_


def make_csv():
    # single_workloads
    data = np.zeros((5, 4))
    """
    for file in single_workloads:
        path = TENSILE_path + file
        path = os.path.join(path, 'repeat_3_result.txt')
        with open(path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 7 or i == 8:
                continue
            assert title[i] in line
            temp = line.replace(title[i] + ':', '')
            mean = float(format(float(temp.split(' ')[0]), '.4f'))
            # std = round(float(line.split(' ')[2]))
            if i == 0:
                col = 0
            elif i == 1:
                col = 1
            elif i == 5:
                col = 3
                mean *= 50
            elif i == 6:
                col = 2
            else:
                continue
            row = get_row(path)
            data[row, col] = mean
    df = pd.DataFrame(data)
    df.index = single_workloads
    df.columns = ['MSR', 'EOR', "CBR", "TIME"]
    df.to_csv(result_csv_log+'SingleWorkloads.csv')
    """
    # multi_workloads
    MSR = np.zeros((15, 4))
    EOR = np.zeros((15, 4))
    TIME = np.zeros((15, 4))
    CBR = np.zeros((15, 4))
    MSR_cold_start = np.zeros((15, 4))
    EOR_cold_start = np.zeros((15, 4))
    CBR_cold_start = np.zeros((15, 4))

    # TENSILE
    for file in multi_workloads:
        path = TENSILE_path + file
        if 'MDW' in file:
            # path = os.path.join(file, 'repeat_10_result.txt')
            continue
        else:
            path = os.path.join(path, 'repeat_5_result.txt')
        with open(path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if i in [0, 1, 5, 6, 9, 10, 12]:
                assert title[i] in line
                temp = line.replace(title[i] + ':', '')
                mean = float(format(float(temp.split(' ')[0]), '.4f'))
                row = get_row(path)
                if 'x1' in path:
                    col = 0
                elif 'x2' in path:
                    col = 1
                elif 'x3' in path:
                    col = 2
                elif 'x4' in path:
                    col = 3
                # elif 'MDW' in path:
                #     col = 3
                else:
                    raise Exception(f'unsupported workload:{path}')
                if i == 0:
                    MSR[row, col] = mean
                elif i == 1:
                    EOR[row, col] = mean
                elif i == 5:
                    TIME[row, col] = mean * 50
                elif i == 6:
                    CBR[row, col] = mean
                elif i == 9:
                    MSR_cold_start[row, col] = mean
                elif i == 10:
                    EOR_cold_start[row, col] = mean
                elif i == 12:
                    CBR_cold_start[row, col] = mean
    # vDNN&Capuchin
    for file in baseline_multi_workloads:
        path = os.path.join(baseline_path, file, 'result.txt')
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                # assert baseline_title[i] in line, line
                if i in [5, 6, 7, 8, 9, 16, 17, 18, 19, 20]:
                    temp = line.replace(baseline_title[i] + ':', '')
                    mean = float(format(float(temp.split(' ')[0]), '.4f'))
                    # todo: MDW
                    # ??????
                    if 'x1' in path:
                        col = 0
                    elif 'x2' in path:
                        col = 1
                    elif 'x3' in path:
                        col = 2
                    elif 'x4' in path:
                        col = 3
                    else:
                        raise Exception(f'unsupported workload:{path}')
                    # vDNN
                    if 5 <= i <= 9:
                        row = get_row(path) + 5
                        if i == 7:
                            MSR[row, col] = mean
                        elif i == 8:
                            EOR[row, col] = mean
                        elif i == 9:
                            CBR[row, col] = mean
                        elif i == 6:
                            TIME[row, col] = mean
                    # Capuchin
                    elif 16 <= i:
                        row = get_row(path) + 10
                        if i == 18:
                            MSR[row, col] = mean
                        elif i == 19:
                            EOR[row, col] = mean
                        elif i == 20:
                            CBR[row, col] = mean
                        elif i == 17:
                            TIME[row, col] = mean
        except Exception:
            pass
    df = pd.DataFrame(MSR)
    df.index = ['TENSILE:' + t for t in single_workloads] + ['vDNN:' + t for t in single_workloads] + ['Capuchin:' + t
                                                                                                       for t in
                                                                                                       single_workloads]
    df.columns = ['x1', 'x2', "x3", "x4"]
    df.to_csv(result_csv_log+'MultiWorkloadsMSR.csv')
    df = pd.DataFrame(EOR)
    df.index = ['TENSILE:' + t for t in single_workloads] + ['vDNN:' + t for t in single_workloads] + ['Capuchin:' + t
                                                                                                       for t in
                                                                                                       single_workloads]
    df.columns = ['x1', 'x2', "x3", "x4"]
    df.to_csv(result_csv_log+'MultiWorkloadsEOR.csv')
    df = pd.DataFrame(CBR)
    df.index = ['TENSILE:' + t for t in single_workloads] + ['vDNN:' + t for t in single_workloads] + ['Capuchin:' + t
                                                                                                       for t in
                                                                                                       single_workloads]
    df.columns = ['x1', 'x2', "x3", "x4"]
    df.to_csv(result_csv_log+'MultiWorkloadsCBR.csv')
    df = pd.DataFrame(MSR_cold_start)
    df.index = ['TENSILE:' + t for t in single_workloads] + ['vDNN:' + t for t in single_workloads] + ['Capuchin:' + t
                                                                                                       for t in
                                                                                                       single_workloads]
    df.columns = ['x1', 'x2', "x3", "x4"]
    df.to_csv(result_csv_log+'MultiWorkloadsMSR_cold_start.csv')
    df = pd.DataFrame(EOR_cold_start)
    df.index = ['TENSILE:' + t for t in single_workloads] + ['vDNN:' + t for t in single_workloads] + ['Capuchin:' + t
                                                                                                       for t in
                                                                                                       single_workloads]
    df.columns = ['x1', 'x2', "x3", "x4"]
    df.to_csv(result_csv_log+'MultiWorkloadsEOR_cold_start.csv')
    df = pd.DataFrame(CBR_cold_start)
    df.index = ['TENSILE:' + t for t in single_workloads] + ['vDNN:' + t for t in single_workloads] + ['Capuchin:' + t
                                                                                                       for t in
                                                                                                       single_workloads]
    df.columns = ['x1', 'x2', "x3", "x4"]
    df.to_csv(result_csv_log+'MultiWorkloadsCBR_cold_start.csv')
    df = pd.DataFrame(TIME)
    df.index = ['TENSILE:' + t for t in single_workloads] + ['vDNN:' + t for t in single_workloads] + ['Capuchin:' + t
                                                                                                       for t in
                                                                                                       single_workloads]
    df.columns = ['x1', 'x2', "x3", "x4"]
    df.to_csv(result_csv_log+'MultiWorkloadsTIME.csv')

    # batch_size
    MSR = np.zeros((5, 5))
    EOR = np.zeros((5, 5))
    CBR = np.zeros((5, 5))
    # MSR[2, 4] = None
    # MSR[4, 4] = None
    # EOR[2, 4] = None
    # EOR[4, 4] = None
    # CBR[2, 4] = None
    # CBR[4, 4] = None
    assert len(batch_size_workloads)%5==0
    for file in batch_size_workloads:
        path = TENSILE_path + file
        path = os.path.join(path, 'repeat_5_result.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                col = batch_size_workloads.index(file)%5
                #col = batch_size_workloads_col[file]
                if i == 0 or i == 1 or i == 6:
                    assert title[i] in line
                    temp = line.replace(title[i] + ':', '')
                    mean = float(format(float(temp.split(' ')[0]), '.4f'))
                    row = get_row(path)
                    if i == 0:
                        MSR[row, col] = mean
                    elif i == 1:
                        EOR[row, col] = mean
                    elif i == 6:
                        CBR[row, col] = mean
    df = pd.DataFrame(MSR)
    df.index = single_workloads
    df.columns = ['4', '8', '16', '32', '64']
    df.to_csv(result_csv_log+'BatchSizeMSR.csv')
    df = pd.DataFrame(EOR)
    df.index = single_workloads
    df.columns = ['4', '8', '16', '32', '64']
    df.to_csv(result_csv_log+'BatchSizeEOR.csv')
    df = pd.DataFrame(CBR)
    df.index = single_workloads
    df.columns = ['4', '8', '16', '32', '64']
    df.to_csv(result_csv_log+'BatchSizeCBR.csv')
