import tool.tools as tools
import time

node_name = ["node1", "node2"]
node_ip = ["192.168.144.101", "192.168.144.102"]
namespace = "sock-shop"
scaling_pod_name = "front-end"
mode = "statuscale"

parameter = tools.IncrementalPID({}, 0).to_dict()
upper_sum_threshold = 38
lower_sum_threshold = -16
upper_sole_threshold = 8
lower_sole_threshold = -5

interval_time = 20
cooling_time = 6 * interval_time
hori_ver_time = 2 * interval_time
safe_time = 2 * interval_time

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
num = 0
while True:
    pod_name = tools.collect_metrics(namespace)
    pod_id = tools.collect_pod_id(pod_name, namespace)
    pod_locate = tools.locate_node(namespace, node_name)
    parameter = tools.scaling(pod_locate, pod_name, scaling_pod_name, mode, namespace, cooling_time,
                              upper_sum_threshold, lower_sum_threshold, upper_sole_threshold, lower_sole_threshold,
                              hori_ver_time, safe_time, parameter)
    num = num + 1
    if num % 5 == 0:
        tools.gbm_train(params)
    time.sleep(interval_time)
