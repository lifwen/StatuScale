import time
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

for k in range(0, 1):
    time_start = time.time()
    df_train = pd.read_csv('train.csv', header=None, sep='\t')
    new_df = pd.read_csv('train.csv')
    datas = new_df.dropna(axis=0, how='any')
    X = datas.iloc[0:39995, 0:5]
    Y = datas.iloc[0:39995, 5:6]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=14)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.2,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'force_col_wise': True
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=20)
    gbm.save_model('model.txt')
    time_end = time.time()
    time_c = time_end - time_start
    print('time cost', time_c, 's')


plt.rcParams["font.family"] = "Times New Roman"
csv = pd.read_csv("test.csv")
res = csv['0']
gbm = lgb.Booster(model_file='model.txt')
x_min_main, x_max_main = -1000, 28000
y_min_main, y_max_main = 18, 105
x_min_zoom, x_max_zoom = 0, 2000
y_min_zoom, y_max_zoom = 15, 50
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(range(27000), res[:27000], linewidth=1)
axins = inset_axes(ax, width='80%', height='20%', loc='upper center')
axins.plot(range(27000), res[:27000], linewidth=1)

for i in range(0, 27000):
    A = [res[i], res[i + 1], res[i + 2], res[i + 3], res[i + 4]]
    df = pd.DataFrame(np.array([A]))
    if gbm.predict(df)[0] + 5 < res[i + 5]:
        ax.scatter(i + 5, res[i + 5], color='r', s=25)
        if x_min_zoom <= i + 5 <= x_max_zoom and y_min_zoom <= res[i + 5] <= y_max_zoom:
            axins.scatter(i + 5, res[i + 5], color='r', s=50)

ax.fill_between([x_min_zoom, x_max_zoom], y_min_zoom, y_max_zoom, color='green', alpha=0.6)
ax.set_xlim(x_min_main, x_max_main)
ax.set_ylim(y_min_main, y_max_main)
ax.set_yticks([y for y in ax.get_yticks() if y <= 100])
axins.set_xlim(x_min_zoom, x_max_zoom)
axins.set_ylim(y_min_zoom, y_max_zoom)
axins.tick_params(axis='both', labelsize=18)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel("Time (10s)", size=22)
ax.set_ylabel("CPU Utilization (%)", size=22)
plt.savefig('figure_2.png', dpi=100,bbox_inches="tight")
plt.show()
