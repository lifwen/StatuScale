import os
import csv
import time
import numpy as np
import subprocess
import lightgbm as lgb
import pandas as pd


def collect_metrics(namespace):
    pod_name = []
    try:
        result = subprocess.check_output("kubectl top pod -n " + namespace, shell=True, text=True)
        top_pod_output = result.strip().split('\n')
        current_time = str(time.time())
        for i in range(1, len(top_pod_output)):
            top_pod_value = top_pod_output[i].split()
            top_pod_value.append(current_time)
            pod_name.append(top_pod_value[0])
            if not os.path.exists("./history/" + top_pod_value[0] + '.csv'):
                with open("./history/" + top_pod_value[0] + '.csv', 'a', newline='') as out:
                    csv_writer = csv.writer(out, dialect='excel')
                    csv_writer.writerow(["POD", "CPU", "MEMORY", "TIME"])
                    csv_writer.writerow(top_pod_value)
            else:
                with open("./history/" + top_pod_value[0] + '.csv', 'a', newline='') as out:
                    csv_writer = csv.writer(out, dialect='excel')
                    csv_writer.writerow(top_pod_value)

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    return pod_name


def collect_pod_id(pod_name, namespace):
    command = ''
    pod_id = []
    for i in range(0, len(pod_name)):
        command_temp = "kubectl get pod -n " + namespace + " " + pod_name[i] + " -o jsonpath='{.metadata.uid}'"
        command = command + command_temp + " ; "
    pod_id_result = subprocess.check_output(command, shell=True)
    pod_id_result = pod_id_result.strip().decode('utf-8')
    for i in range(0, len(pod_id_result), 36):
        pod_id.append(pod_id_result[i:i + 8])
    return pod_id


def locate_node(namespace, node_name):
    pod_locate = []
    for i in range(0, len(node_name)):
        pod_locate.append([node_name[i]])
    for i in range(0, len(node_name)):
        try:
            pod_node = subprocess.check_output(
                "kubectl get pod -n " + namespace + " -o wide | grep '" + pod_locate[i][0] + "'", shell=True, text=True)
            pod_node = pod_node.strip().split('\n')
            for line in pod_node:
                line = line.replace(',', '\t')
                list0 = line.split()
                pod_locate[i].append(list0[0])
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
    return pod_locate


def maxfilesize(replicas_name, base_path):
    max_file_size = 0
    max_file_path = ''
    for file_name in replicas_name:
        file_path = os.path.join(base_path, file_name + ".csv")
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
        else:
            file_size = 0
        if file_size > max_file_size:
            max_file_size = file_size
            max_file_path = file_path
    return max_file_path


def scaling(pod_locate, pod_name, scaling_pod_name, mode, namespace, cooling_time, upper_sum_threshold,
            lower_sum_threshold, upper_sole_threshold, lower_sole_threshold, hori_ver_time, safe_time, parameter):
    replicas_name = []
    replicas_locate = []
    replicas_id = []
    try:
        all_replicas = subprocess.check_output(
            "kubectl get pod -n " + namespace + " -o wide | grep " + scaling_pod_name + " | grep Running",
            shell=True, text=True)
        all_replicas = all_replicas.strip().split('\n')
        for line in all_replicas:
            line = line.replace(',', '\t')
            list0 = line.split()
            replicas_name.append(list0[0])
            replicas_locate.append(list0[6])
            try:
                all_replicas_id = subprocess.check_output(
                    "kubectl get pod -n " + namespace + " " + list0[0] + " -o jsonpath='{.metadata.uid}'",
                    shell=True, text=True)
                all_replicas_id = all_replicas_id.strip().split('\n')
                for element in all_replicas_id:
                    replicas_id.append(element)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    history_file = maxfilesize(replicas_name, './history/')

    tag_title = True
    memory_usage = []
    cpu_usage = []
    timestamp = []
    with open(history_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if tag_title:
                tag_title = False
                continue
            cpu_usage.append(float(row[1][:-1]))
            memory_usage.append(float(row[2][:-2]))
            timestamp.append(float(row[3]))

    if os.path.exists("./scale/front-end.csv"):
        with open("./scale/front-end.csv", 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            scale_info = list(csv_reader)[-1]
    else:
        with open("./scale/front-end.csv", 'a', newline='') as out:
            csv_writer = csv.writer(out, dialect='excel')
            csv_writer.writerow(["Resource", "Cooling Time", "Current Time", "Replicas", "Mode"])
            csv_writer.writerow([0, 0, 0, 0, 0])
            scale_info = [0, 0, 0, 0, 0]

    # os.path.join('./scale/',os.path.basename(history_file))
    if mode == "statuscale":
        parameter = statuscale(cpu_usage, memory_usage, timestamp, replicas_name, replicas_locate,
                               replicas_id,
                               scale_info, "./scale/front-end.csv",
                               cooling_time,
                               namespace, upper_sum_threshold, lower_sum_threshold,
                               upper_sole_threshold,
                               lower_sole_threshold, hori_ver_time, safe_time, parameter)
    elif mode == "showar":
        parameter = showar(cpu_usage, memory_usage, timestamp, replicas_name, replicas_locate,
                           replicas_id,
                           scale_info, os.path.join('./scale/', os.path.basename(history_file)),
                           cooling_time,
                           namespace, parameter)
    elif mode == "hyscale":
        hyscale(cpu_usage, memory_usage, timestamp, replicas_name, replicas_locate, replicas_id,
                scale_info, "./scale/front-end.csv", cooling_time,
                namespace)
    elif mode == "gbmscaler":
        gbmscaler(cpu_usage, memory_usage, timestamp, replicas_name, replicas_locate, replicas_id,
                  scale_info,"./scale/front-end.csv", cooling_time,
                  namespace)
    return parameter


def gbm_train(params):
    new_df = pd.read_csv('./train_data/train.csv')
    datas = new_df.dropna(axis=0, how='any')
    X = datas.iloc[:, 0:5]
    Y = datas.iloc[:, 5:6]
    lgb_train = lgb.Dataset(X, Y)
    gbm = lgb.train(params, lgb_train, num_boost_round=20)
    gbm.save_model('./train_data/model.txt')


def statuscale(cpu_usage, memory_usage, timestamp, replicas_name, replicas_locate, replicas_id, scale_info, scale_file,
               cooling_time, namespace, upper_sum_threshold, lower_sum_threshold, upper_sole_threshold,
               lower_sole_threshold, hori_ver_time, safe_time, parameter):
    if not os.path.exists("./scale/horizontal.csv"):
        with open("./scale/horizontal.csv", 'a', newline='') as out:
            csv_writer = csv.writer(out, dialect='excel')
            csv_writer.writerow([0, 0, 0, 0, 0, 0])

    with open("./scale/horizontal.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        row = list(reader)[-1]
        if float(cpu_usage[-1]) < 15:
            with open("./scale/horizontal.csv", "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                row1 = [cpu_usage[-1], float(row[1]), float(row[3]), float(row[4]),
                        -(1.7 ** ((30 - float(cpu_usage[-1] / 10)) / 2.5) / 10), row[-1]]

                # if float(row1[1]) + float(cooling_time) < float(timestamp[-1]) and (
                #         float(row1[2]) + float(row1[3]) + float(row1[4])) < lower_sum_threshold and float(
                #     row1[4]) < lower_sole_threshold and len(replicas_name) > 1:
                #     print("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                #         max(len(replicas_name) - 1, 1)) + "  -n " + namespace)
                #     os.system("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                #         max(len(replicas_name) - 1, 1)) + "  -n " + namespace)
                #     row1 = [cpu_usage[-1], str(time.time()), 0, 0, 0, -1]
                # csv_writer.writerow(row1)

        if float(cpu_usage[-1]) > 15:
            with open("./scale/horizontal.csv", "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                row1 = [cpu_usage[-1], float(row[1]), float(row[3]), float(row[4]),
                        1.7 ** (float(cpu_usage[-1] / 10) / 2.5) / 10, row[-1]]

                # if float(row1[1]) + float(cooling_time) < float(timestamp[-1]) and (
                #         (float(row1[2]) + float(row1[3]) + float(row1[4])) > upper_sum_threshold) and (
                #         float(row1[4]) > upper_sole_threshold):
                #     print("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                #         len(replicas_name) + 1) + "  -n " + namespace)
                #     os.system("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                #         len(replicas_name) + 1) + "  -n " + namespace)
                #     row1 = [cpu_usage[-1], str(time.time()), 0, 0, 0, 1]
                # csv_writer.writerow(row1)

        if float(row1[1]) + hori_ver_time > float(time.time()) and str(row1[5]) == "1":
            with open(scale_file, 'r') as csvfile1:
                reader = csv.reader(csvfile1)
                scale_info_last = list(reader)[-1]
            row2 = [max(float(scale_info_last[0]), cpu_usage[-1] * 1.15) * 0.9, row[1]]
            for i in range(0, len(replicas_id)):
                print("horizontal vertical: ssh root@" + str(replicas_locate[i]) + " 'python scale.py " + str(
                    min(max(int(row2[0]) * 100, 1000), 30000))
                      + " " + str(replicas_id[i]) + "'")
                os.system("ssh root@" + str(replicas_locate[i]) + " 'python scale.py " + str(
                    min(max(int(row2[0]) * 100, 1000), 30000))
                          + " " + str(replicas_id[i]) + "'")
            with open(scale_file, "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(
                    [row2[0], row[1], timestamp[-1], len(replicas_name), "horizontal_vertical_down", timestamp[-1],
                     '', '', 1 if len(scale_info_last) < 8 else scale_info_last[8]])
            return parameter

        if float(row1[1]) + hori_ver_time > float(time.time()) and str(row1[5]) == "-1":
            with open(scale_file, 'r') as csvfile1:
                reader = csv.reader(csvfile1)
                scale_info_last = list(reader)[-1]
            row2 = [float(scale_info_last[0]) * 1.1, row[1]]
            for i in range(0, len(replicas_id)):
                print("horizontal vertical: ssh root@" + str(replicas_locate[i]) + " 'python scale.py " + str(
                    min(max(int(row2[0]) * 100, 1000), 30000)) + " " + str(replicas_id[i]) + "'")
                os.system("ssh root@" + str(replicas_locate[i]) + " 'python scale.py " + str(
                    min(max(int(row2[0]) * 100, 1000), 30000)) + " " + str(replicas_id[i]) + "'")
            with open(scale_file, "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(
                    [row2[0], row[1], timestamp[-1], len(replicas_name), "horizontal_vertical_up",
                     timestamp[-1], '', '', 1 if len(scale_info_last) < 8 else scale_info_last[8]])
            return parameter

    with open(scale_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        scale_info_last = list(reader)[-1]

        tag3 = len(cpu_usage) > 5
        tag4 = scale_info_last[4] == "horizontal_vertical_up" or scale_info_last[
            4] == "horizontal_vertical_down"
        tag1 = len(scale_info_last) > 5 and ((
                                                     scale_info_last[4] == "threshold" or scale_info_last[
                                                 4] == "pid") and float(scale_info_last[5]) < float(timestamp[-1]))

        tag2 = scale_info_last[4] == "predict" and cpu_usage[-1] < float(scale_info_last[6]) * timestamp[-1] + float(
            scale_info_last[7])

        if tag3 and not tag4 and (tag1 or tag2):

            if scale_info_last[4] == "threshold" or scale_info_last[4] == "pid" or len(cpu_usage) % 10 == 0:
                start = len(timestamp) - 5
            else:
                start = int(scale_info_last[8])
            [k, b] = np.polyfit(timestamp[start:], cpu_usage[start:], deg=1)

            gbm = lgb.Booster(model_file='./train_data/model.txt')
            df = pd.DataFrame(np.array([[cpu_usage[-5] / 3, cpu_usage[-4] / 3, cpu_usage[-3] / 3,
                                         cpu_usage[-2] / 3, cpu_usage[-1] / 3]]))
            y_pred = gbm.predict(df)[0] * 3 / 0.8

            for i in range(0, len(replicas_id)):
                print("predict ssh root@" + str(replicas_locate[i]) + " 'python scale.py " + str(
                    min(max(int(y_pred) * 100, 1000), 30000)) + " " + str(
                    replicas_id[i]) + "'")
                os.system("ssh root@" + str(replicas_locate[i]) + " 'python scale.py " + str(
                    min(max(int(y_pred) * 100, 1000), 30000)) + " " + str(
                    replicas_id[i]) + "'")
            row = [y_pred, scale_info_last[1], timestamp[-1], scale_info_last[3], "predict",
                   timestamp[-1], k,
                   b + 30 * np.std(cpu_usage[start:]) / np.mean(cpu_usage[start:]) + 7, start]
            with open(scale_file, "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(row)
        else:

            if not tag3:
                row = [cpu_usage[-1] / 0.8, scale_info_last[1], timestamp[-1], scale_info_last[3], "threshold",
                       timestamp[-1], '', '', 1]
            else:
                pid_result = pid_output(parameter, cpu_usage)
                parameter = pid_result[1]
                row = [pid_result[0], scale_info_last[1], timestamp[-1], scale_info_last[3], "pid",
                       scale_info_last[5] if float(scale_info_last[5]) > float(timestamp[-1]) else timestamp[
                                                                                                       -1] + safe_time,
                       '', '', scale_info_last[8]]

            for i in range(0, len(replicas_id)):
                print("threshold or pid: ssh root@" + str(replicas_locate[i]) + " 'python scale.py " + str(
                    min(max(int(row[0]) * 100, 1000), 30000)) + " " + str(replicas_id[i]) + "'")
                os.system("ssh root@" + str(replicas_locate[i]) + " 'python scale.py " + str(
                    min(max(int(row[0]) * 100, 1000), 30000)) + " " + str(replicas_id[i]) + "'")
            with open(scale_file, "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(row)

    if len(cpu_usage) > 6:
        with open("./train_data/train.csv", "a", newline='') as out:
            csv_writer = csv.writer(out, dialect="excel")
            csv_writer.writerow(
                [cpu_usage[-6] / 3, cpu_usage[-5] / 3, cpu_usage[-4] / 3, cpu_usage[-3] / 3, cpu_usage[-2] / 3,
                 cpu_usage[-1] / 3])

    return parameter


def pid_output(parameter, cpu_usage):
    parameter['y'] = cpu_usage[-1] / 300
    parameter['y1'] = cpu_usage[-2] / 300
    parameter['y2'] = cpu_usage[-3] / 300
    Process1 = IncrementalPID(parameter, 1)
    try:
        for i in range(0, 10):
            Process1.SetStepSignal(0.8)
        parameter = Process1.to_dict()
        rate = Process1.y
    except:
        parameter = IncrementalPID({}, 0).to_dict()
        rate = 0.8
    return [(cpu_usage[-1] + cpu_usage[-2] + cpu_usage[-3]) / 3 / rate, parameter]


def showar(cpu_usage, memory_usage, timestamp, replicas_name, replicas_locate, replicas_id, scale_info, scale_file,
           cooling_time, namespace, parameter):
    if len(cpu_usage) == 1:
        row = [cpu_usage[-1] / 0.8, timestamp[-1], timestamp[-1], "vertical"]
    else:
        if len(cpu_usage) <= 5:
            row = [cpu_usage[-1] / 0.8, scale_info[1], timestamp[-1], "vertical"]
        else:
            y_pred = (sum(cpu_usage[-5:]) / len(cpu_usage[-5:]) + 3 * np.std(cpu_usage[-5:])) * 1.05
            pid_result = pid_output(parameter, cpu_usage)
            parameter = pid_result[1]
            if len(replicas_name) > 1 and pid_result[0] <= (len(replicas_name) - 1) * 300 / (
                    len(replicas_name)) and float(scale_info[1]) + int(cooling_time) < float(timestamp[-1]):
                row = [y_pred, timestamp[-1], timestamp[-1], max(len(replicas_name) - 1, 1), "horizontal"]
                print("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                    row[3]) + "  -n " + namespace)
                os.system("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                    row[3]) + "  -n " + namespace)
            else:
                if pid_result[0] >= 300 and float(scale_info[1]) + float(
                        cooling_time) < float(timestamp[-1]):
                    row = [y_pred, timestamp[-1], timestamp[-1], max(len(replicas_name) + 1, 1), "horizontal"]
                    print("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                        row[3]) + "  -n " + namespace)
                    os.system("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                        row[3]) + "  -n " + namespace)
                else:
                    row = [y_pred, scale_info[1], timestamp[-1], "", "vertical"]

    for i in range(0, len(replicas_id)):
        print("ssh root@" + replicas_locate[i] + " 'python scale.py " + str(
            min(max(int(row[0]) * 100, 1000), 30000)) + " " + str(
            replicas_id[i]) + "'")
        os.system("ssh root@" + replicas_locate[i] + " 'python scale.py " + str(
            min(max(int(row[0]) * 100, 1000), 30000)) + " " + str(
            replicas_id[i]) + "'")
    with open(scale_file, "a", newline='') as out:
        csv_writer = csv.writer(out, dialect="excel")
        csv_writer.writerow(row)
    return parameter


def hyscale(cpu_usage, memory_usage, timestamp, replicas_name, replicas_locate, replicas_id, scale_info, scale_file,
            cooling_time, namespace):
    if len(cpu_usage) == 1:
        row = [cpu_usage[-1] / 0.82, timestamp[-1], timestamp[-1], "", "vertical"]
    else:
        if len(cpu_usage) <= 5:
            row = [cpu_usage[-1] / 0.82, scale_info[1], timestamp[-1], "", "vertical"]
        else:
            if len(replicas_name) > 1 and sum(cpu_usage[-5:]) / 4 <= (len(replicas_name) - 1) * 30 / (  # 300
                    len(replicas_name)) and float(scale_info[1]) + int(cooling_time) < float(timestamp[-1]):
                row = [sum(cpu_usage[-5:]) / 5/0.82, timestamp[-1], timestamp[-1], max(len(replicas_name) - 1, 1),
                       "horizontal"]
                print("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                    row[3]) + "  -n " + namespace)
                os.system("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                    row[3]) + "  -n " + namespace)
            else:
                if sum(cpu_usage[-5:]) / 4 >= 300 and float(scale_info[1]) + int(
                        cooling_time) < float(timestamp[-1]):
                    row = [sum(cpu_usage[-5:]) /5/ 0.82, timestamp[-1], timestamp[-1], max(len(replicas_name) + 1, 1),
                           "horizontal"]
                    print("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                        row[3]) + "  -n " + namespace)
                    os.system("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                        row[3]) + "  -n " + namespace)
                else:
                    row = [sum(cpu_usage[-5:]) /5/ 0.82, scale_info[1], timestamp[-1], "", "vertical"]

    for i in range(0, len(replicas_id)):
        print("ssh root@" + replicas_locate[i] + " 'python scale.py " + str(
            min(max(int(row[0]) * 100, 1000), 30000)) + " " + str(
            replicas_id[i]) + "'")
        os.system("ssh root@" + replicas_locate[i] + " 'python scale.py " + str(
            min(max(int(row[0]) * 100, 1000), 30000)) + " " + str(
            replicas_id[i]) + "'")
    with open(scale_file, "a", newline='') as out:
        csv_writer = csv.writer(out, dialect="excel")
        csv_writer.writerow(row)


def gbmscaler(cpu_usage, memory_usage, timestamp, replicas_name, replicas_locate, replicas_id, scale_info, scale_file,
              cooling_time, namespace):
    if len(cpu_usage) == 1:
        row = [cpu_usage[-1] / 0.8, timestamp[-1], timestamp[-1], "", "vertical"]
    else:
        if len(cpu_usage) <= 5:
            row = [cpu_usage[-1] / 0.8, scale_info[1], timestamp[-1], "", "vertical"]
        else:
            gbm = lgb.Booster(model_file='./train_data/model.txt')
            df = pd.DataFrame(np.array([[cpu_usage[-5] / 3, cpu_usage[-4] / 3,
                                         cpu_usage[-3] / 3, cpu_usage[-2] / 3,
                                         cpu_usage[-1] / 3]]))
            y_pred = gbm.predict(df)[0] / 0.8 * 3
            if len(replicas_name) > 123456 and y_pred <= (len(replicas_name) - 1) * 300 / ( # 1
                    len(replicas_name)) and float(scale_info[1]) + int(cooling_time) < float(timestamp[-1]):
                row = [y_pred, timestamp[-1], timestamp[-1], max(len(replicas_name) - 1, 1), "horizontal"]
                print("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                    row[3]) + "  -n " + namespace)
                os.system("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                    row[3]) + "  -n " + namespace)
            else:
                if y_pred >= 300123456 and float(scale_info[1]) + int( # 300
                        cooling_time) < float(timestamp[-1]):
                    row = [y_pred, timestamp[-1], timestamp[-1], max(len(replicas_name) + 1, 1), "horizontal"]
                    print("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                        row[3]) + "  -n " + namespace)
                    os.system("kubectl scale deployment " + replicas_name[0][:-17] + " --replicas=" + str(
                        row[3]) + "  -n " + namespace)
                else:
                    row = [y_pred, scale_info[1], timestamp[-1], "", "vertical"]

    for i in range(0, len(replicas_id)):
        print("ssh root@" + replicas_locate[i] + " 'python scale.py " + str(
            min(max(int(row[0]) * 100, 1000), 30000)) + " " + str(
            replicas_id[i]) + "'")
        os.system("ssh root@" + replicas_locate[i] + " 'python scale.py " + str(
            min(max(int(row[0]) * 100, 1000), 30000)) + " " + str(
            replicas_id[i]) + "'")
    with open(scale_file, "a", newline='') as out:
        csv_writer = csv.writer(out, dialect="excel")
        csv_writer.writerow(row)
    if len(cpu_usage) > 6:
        with open("./train_data/train.csv", "a", newline='') as out:
            csv_writer = csv.writer(out, dialect="excel")
            csv_writer.writerow(
                [cpu_usage[-6] / 3, cpu_usage[-5] / 3, cpu_usage[-4] / 3, cpu_usage[-3] / 3, cpu_usage[-2] / 3,
                 cpu_usage[-1] / 3])


class IncrementalPID:
    def __init__(self, saved_data, tag):
        if tag == 1:
            self.xite_1 = saved_data['xite_1']
            self.alfa = saved_data['alfa']
            self.IN = saved_data['IN']
            self.H = saved_data['H']
            self.Out = saved_data['Out']
            self.wi = saved_data['wi']
            self.wi_1 = saved_data['wi_1']
            self.wi_2 = saved_data['wi_2']
            self.wi_3 = saved_data['wi_3']
            self.wo = saved_data['wo']
            self.wo_1 = saved_data['wo_1']
            self.wo_2 = saved_data['wo_2']
            self.wo_3 = saved_data['wo_3']
            self.Kp = saved_data['Kp']
            self.Ki = saved_data['Ki']
            self.Kd = saved_data['Kd']
            self.x = saved_data['x']
            self.y = saved_data['y']
            self.y_1 = saved_data['y_1']
            self.y_2 = saved_data['y_2']
            self.e = saved_data['e']
            self.e_1 = saved_data['e_1']
            self.e_2 = saved_data['e_2']
            self.de_1 = saved_data['de_1']
            self.u = saved_data['u']
            self.u_1 = saved_data['u_1']
            self.u_2 = saved_data['u_2']
            self.u_3 = saved_data['u_3']
            self.u_4 = saved_data['u_4']
            self.u_5 = saved_data['u_5']
            self.Oh = saved_data['Oh']
            self.I = saved_data['I']
            self.Oh_sub = saved_data['Oh_sub']
            self.K_sub = saved_data['K_sub']
            self.dK_sub = saved_data['dK_sub']
            self.delta3_sub = saved_data['delta3_sub']
            self.dO_sub = saved_data['dO_sub']
            self.delta2_sub = saved_data['delta2_sub']
            self.den = saved_data['den']
            self.num = saved_data['num']
            self.du = saved_data['du']
        else:
            self.xite_1 = 0.2
            self.alfa = 0.95
            self.IN = 4
            self.H = 5
            self.Out = 3
            self.wi = np.mat([[-0.6394, -0.2696, -0.3756, -0.7023],
                              [-0.8603, -0.2013, -0.5024, -0.2596],
                              [-1.0000, 0.5543, -1.0000, -0.5437],
                              [-0.3625, -0.0724, 0.6463, -0.2859],
                              [0.1425, 0.0279, -0.5406, -0.7660]]
                             )
            self.wi_1 = self.wi
            self.wi_2 = self.wi
            self.wi_3 = self.wi
            self.wo = np.mat([[0.7576, 0.2616, 0.5820, -0.1416, -0.1325],
                              [-0.1146, 0.2949, 0.8352, 0.2205, 0.4508],
                              [0.7201, 0.4566, 0.7672, 0.4962, 0.3632]]
                             )
            self.wo_1 = self.wo
            self.wo_2 = self.wo
            self.wo_3 = self.wo
            self.Kp = 0.0
            self.Ki = 0.0
            self.Kd = 0.0
            self.x = [self.Kp, self.Ki, self.Kd]
            self.y = 0.0
            self.y_1 = 0.0
            self.y_2 = 0.0
            self.e = 0.0
            self.e_1 = 0.0
            self.e_2 = 0.0
            self.de_1 = 0.0
            self.u = 0.0
            self.u_1 = 0.0
            self.u_2 = 0.0
            self.u_3 = 0.0
            self.u_4 = 0.0
            self.u_5 = 0.0
            self.Oh = np.mat(np.zeros((self.H, 1)))
            self.I = self.Oh
            self.Oh_sub = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.K_sub = [0.0, 0.0, 0.0]
            self.dK_sub = [0.0, 0.0, 0.0]
            self.delta3_sub = [0.0, 0.0, 0.0]
            self.dO_sub = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.delta2_sub = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.den = -0.8251
            self.num = 0.2099
            self.du = 0.0

    def to_dict(self):
        return {
            'xite_1': self.xite_1,
            'alfa': self.alfa,
            'IN': self.IN,
            'H': self.H,
            'Out': self.Out,
            'wi': self.wi,
            'wi_1': self.wi_1,
            'wi_2': self.wi_2,
            'wi_3': self.wi_3,
            'wo': self.wo,
            'wo_1': self.wo_1,
            'wo_2': self.wo_2,
            'wo_3': self.wo_3,
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
            'x': self.x,
            'y': self.y,
            'y_1': self.y_1,
            'y_2': self.y_2,
            'e': self.e,
            'e_1': self.e_1,
            'e_2': self.e_2,
            'de_1': self.de_1,
            'u': self.u,
            'u_1': self.u_1,
            'u_2': self.u_2,
            'u_3': self.u_3,
            'u_4': self.u_4,
            'u_5': self.u_5,
            'Oh': self.Oh,
            'I': self.I,
            'Oh_sub': self.Oh_sub,
            'K_sub': self.K_sub,
            'dK_sub': self.dK_sub,
            'delta3_sub': self.delta3_sub,
            'dO_sub': self.dO_sub,
            'delta2_sub': self.delta2_sub,
            'den': self.den,
            'num': self.num,
            'du': self.du
        }

    def SetStepSignal(self, SetSignal):
        self.y = self.y + self.du
        self.e = SetSignal - self.y
        self.xi = np.mat([SetSignal, self.y, self.e, 5])
        self.x[0] = self.e - self.e_1
        self.x[1] = self.e
        self.x[2] = self.e - 2 * self.e_1 + self.e_2
        self.epid = np.mat([[self.x[0]], [self.x[1]], [self.x[2]]])
        self.I = np.dot(self.xi, (self.wi.T))
        for i1 in range(5):
            self.Oh_sub[i1] = (np.e ** (self.I.tolist()[0][i1]) - np.e ** (-self.I.tolist()[0][i1])) / (
                    np.e ** (self.I.tolist()[0][i1]) + np.e ** (-self.I.tolist()[0][i1]))
        self.Oh = np.mat([[self.Oh_sub[0]], [self.Oh_sub[1]], [self.Oh_sub[2]], [self.Oh_sub[3]], [self.Oh_sub[4]]])
        self.K = np.dot(self.wo, self.Oh)
        for i2 in range(3):
            self.K_sub[i2] = (np.e ** (self.K.tolist()[i2][0])) / (
                    np.e ** (self.K.tolist()[i2][0]) + np.e ** (-self.K.tolist()[i2][0]))
        self.K = np.mat([[self.K_sub[0]], [self.K_sub[1]], [self.K_sub[2]]])
        self.Kp = self.K_sub[0]
        self.Ki = self.K_sub[1]
        self.Kd = self.K_sub[2]
        self.Kpid = np.mat([self.Kp, self.Ki, self.Kd])
        self.du = np.dot(self.Kpid, self.epid).tolist()[0][0]
        self.u = self.u_1 + self.du
        self.de = self.e - self.e_1
        if self.de > (self.de_1 * 1.04):
            self.xite = 0.7 * self.xite_1
        elif self.de < self.de_1:
            self.xite = 1.05 * self.xite_1
        else:
            self.xite = self.xite_1
        self.dyu = np.sin((self.y - self.y_1) / (self.u - self.u_1 + 0.0000001))
        for i3 in range(3):
            self.dK_sub[i3] = 2 / ((np.e ** (self.K_sub[i3]) + np.e ** (-self.K_sub[i3])) * (
                    np.e ** (self.K_sub[i3]) + np.e ** (-self.K_sub[i3])))
        self.dK = np.mat([self.dK_sub[0], self.dK_sub[1], self.dK_sub[2]])
        for i4 in range(3):
            self.delta3_sub[i4] = self.e * self.dyu * self.epid.tolist()[i4][0] * self.dK_sub[i4]
        self.delta3 = np.mat([self.delta3_sub[0], self.delta3_sub[1], self.delta3_sub[2]])
        for l in range(3):
            for i5 in range(5):
                self.d_wo = (1 - self.alfa) * self.xite * self.delta3_sub[l] * self.Oh.tolist()[i5][0] + self.alfa * (
                        self.wo_1 - self.wo_2)
        self.wo = self.wo_1 + self.d_wo
        for i6 in range(5):
            self.dO_sub[i6] = 4 / ((np.e ** (self.I.tolist()[0][i6]) + np.e ** (-self.I.tolist()[0][i6])) * (
                    np.e ** (self.I.tolist()[0][i6]) + np.e ** (-self.I.tolist()[0][i6])))
        self.dO = np.mat([self.dO_sub[0], self.dO_sub[1], self.dO_sub[2], self.dO_sub[3], self.dO_sub[4]])
        self.segma = np.dot(self.delta3, self.wo)
        for i7 in range(5):
            self.delta2_sub[i7] = self.dO_sub[i7] * self.segma.tolist()[0][i7]
        self.delta2 = np.mat(
            [self.delta2_sub[0], self.delta2_sub[1], self.delta2_sub[2], self.delta2_sub[3], self.delta2_sub[4]])
        self.d_wi = (1 - self.alfa) * self.xite * self.delta2.T * self.xi + self.alfa * (self.wi_1 - self.wi_2)
        self.wi = self.wi_1 + self.d_wi
        self.u_5 = self.u_4
        self.u_4 = self.u_3
        self.u_3 = self.u_2
        self.u_2 = self.u_1
        self.u_1 = self.u
        self.y_2 = self.y_1
        self.y_1 = self.y
        self.wo_3 = self.wo_2
        self.wo_2 = self.wo_1
        self.wo_1 = self.wo
        self.wi_3 = self.wi_2
        self.wi_2 = self.wi_1
        self.wi_1 = self.wi
        self.e_2 = self.e_1
        self.e_1 = self.e
        self.xite_1 = self.xite
