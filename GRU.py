#測試
#套件部分
import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam,SGD
import random
import pandas as pd
import datetime
from keras.callbacks import EarlyStopping
#讀入檔案
# 設定路徑
data_dir = '/home/asis/PM2.5_GPU'
fname = os.path.join(data_dir, 'data105106107_taoyuan.csv')
# 讀檔
f = open(fname)
data = f.read()
f.close()
# 以分隔符號(換行)，分割一列一列資料
lines = data.split('\n')
# 以逗號，分割一字一字資料，針對第 0 列資料
header = lines[0].split(',')
# 第一列資料
line = lines[1:]
#將NA轉乘nan
for i in range(0, len(line)):
    line[i] =["0" if x =='NA' else x for x in line[i].split(',')]
    
float_data = np.zeros((len(line),42))
for i in range(0, len(line)):
    try:
        float_data[i] = [float(x) for x in line[i]]
    except:
        print(line[i])
#在實際使用之前，我們需要將資料先行標準化。這裡的標準化指的是常態標準化。
# 先算出平均數
# axis=0 是指針對每一欄
mean = np.nanmean(float_data[:8783,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,41)],axis=0)
# 中心化
float_data[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,41)] -= mean
# 算出標準差
std = np.nanstd(float_data[:8783,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,41)],axis=0)
# 
float_data[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,41)] /= std

# 真正使用的產生器
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    
    # 無窮迴圈
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][41]
        yield samples, targets  # 不是用 return
# 重新產生資料產生器
DOE_result = pd.DataFrame(pd.read_csv('/home/asis/PM2.5_GPU/DOE/DOE_result_GRU.csv'))
time=[]
Training_loss=[]
Training_mae=[]
validation_loss=[]
validation_mae=[]
Test_loss=[]
Test_mae=[]
result_epoch=[]
#for i in range(0,2):
for i in DOE_result.index:
    for j in DOE_result.columns:
        #print(j)
        locals()[j]=DOE_result[j][i]
        
    delay = 24
    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=8783,
                          shuffle=True,
                          step=step, 
                          batch_size=batchSize)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=8784,
                        max_index=17543,
                        step=step,
                        batch_size=batchSize)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=17544,
                         max_index=None,
                         step=step,
                         batch_size=batchSize)

    # 計算確認資料需產生的次數
    val_steps = (17543 - 8784 - lookback) // batchSize
    # 計算測試資料需產生的次數
    test_steps = (len(float_data) - 17544 - lookback) // batchSize

    #建立儲存list
    

    #建立模型
    starttime = datetime.datetime.now()
    model = Sequential()
    model.add(layers.GRU(unit1,
                         dropout=drop1,
                         recurrent_dropout=dropRnn1,
                         input_shape=(None, float_data.shape[-1]),
                         activation=act1))
    model.add(layers.GRU(unit2,
                           dropout=drop2,
                           recurrent_dropout=dropRnn2,
                           activation=act2))
    model.add(layers.Dense(1))
    model.summary()
    # 開始訓練
    callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=3,restore_best_weights=True) #設定提早停止
    model.compile(optimizer=optimizer,loss="mse",metrics=['mae'])
    history = model.fit_generator(train_gen,
                                  steps_per_epoch= 8784 // batchSize,
                                  epochs=200,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=[callback])
    result_epoch.append(len(history.history["loss"]))
    endtime = datetime.datetime.now()
    total_time=(endtime - starttime).seconds
    time.append(total_time)
    #print (str(total_time)+'秒')
    #結果
    #print('\nTraining ------------')
    #loss = history.history['loss']
    loss=history.history['loss'][-1]*std[17]
    val_loss=history.history['val_loss'][-1]*std[17]
    #print('loss平均',(np.mean(loss)*std[17]))
    #mae = history.history['mae']
    mae=history.history['mae'][-1]*std[17]
    val_mae=history.history['val_mae'][-1]*std[17]
    #print('mae平均',(np.mean(mae)*std[17]))
    Training_loss.append(loss)
    Training_mae.append(mae)
    validation_loss.append(val_loss)
    validation_mae.append(val_mae)
    #print('\nTesting ------------')
    cost = np.array(model.evaluate_generator(test_gen,
                                 steps=test_steps))
    cost_orig=cost*std[17]
    #print('test cost:', (cost*std[17]))
    Test_loss.append(cost_orig[0])
    Test_mae.append(cost_orig[1])

result = pd.DataFrame({'time':time,'Training_loss':Training_loss,'Training_mae':Training_mae
                      ,'validation_loss':validation_loss,'validation_mae':validation_mae,'Test_loss':Test_loss
                      ,'Test_mae':Test_mae,'result_epoch':result_epoch})