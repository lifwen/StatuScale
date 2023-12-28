import csv
import numpy
import time
import sys
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from matplotlib import pyplot as plt
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6, 5))
cpu0 = [0.5, 0.66, 0.75, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 1]
time0 = [33.812,57.794,74.07,83.43,114.166,168.516,228.48,270.54,394.44,468.22,470.46]
x0 = numpy.linspace(time0[0], time0[-1], 5000)
times = numpy.array(time0)
cpus = numpy.array(cpu0)
model = make_interp_spline(times, cpus)
y0 = model(x0)
plt.plot(x0, y0,'-',  label="StatuScale",linewidth=1.5)
print("StatuScale",(1-model(200))*100)
print("StatuScale",(1-model(250))*100)
time0 = [37.528,61.782,77.476,86.96,127.42,199.272,262.3,310.66,448.84,526.66,527.74]
x0 = numpy.linspace(time0[0], time0[-1], 5000)
times = numpy.array(time0)
cpus = numpy.array(cpu0)
model = make_interp_spline(times, cpus)
y0 = model(x0)
plt.plot(x0, y0,':', label="GBMScaler",linewidth=1.5)
print("LightGBM",(1-model(200))*100)
print("LightGBM",(1-model(250))*100)
time0 = [42.698,65.92,81.86,91.568,127.574,179.72,236.28,281.6,413.5,492.86,493.34]
x0 = numpy.linspace(time0[0], time0[-1], 5000)
times = numpy.array(time0)
cpus = numpy.array(cpu0)
model = make_interp_spline(times, cpus)
y0 = model(x0)
plt.plot(x0, y0, '-.',label="Showar",linewidth=1.5)
print("showar",(1-model(200))*100)
print("showar",(1-model(250))*100)
time0 = [42.374,66.266,83.022,94.596,137.234,203.976,268.96,316.62,452.18,533.98,534.96]
x0 = numpy.linspace(time0[0], time0[-1], 5000)
times = numpy.array(time0)
cpus = numpy.array(cpu0)
model = make_interp_spline(times, cpus)
y0 = model(x0)
plt.plot(x0, y0,'--',label="Hyscale",linewidth=1.5)
print("Hyscale",(1-model(200))*100)
print("Hyscale",(1-model(250))*100)
# x = [0, 17.6146789, 228.9908257, 246.6055046, 352.293578, 369.9082569, 422.7522936, 440.3669725, 669.3577982,
#      686.9724771, 1092.110092, 1109.724771, 1391.559633, 1409.174312, 1497.247706, 1514.862385, 1831.926606,
#      1849.541284]
# y = [30.1, 31, 31.5, 29.8, 32.5, 32.5, 37.7, 34.6, 35.8, 37.3, 32.3, 32.5, 40, 36.7, 37.2, 33.6, 36, 28.8]
# # plt.scatter(x, y, s=20, c="r", linewidths=2, alpha=1, marker='o')
# y = [29.3, 29.8, 31.4, 29.8, 33.6, 32.9, 29.7, 31, 37.7, 40.8, 36.3, 32.7, 28.2, 33.2, 33.7, 37, 35.2, 33.2, 31.6, 33.1,
#      33.3, 28.8]
# x = [35.2293578, 123.3027523, 211.3761468, 264.2201835, 387.5229358, 457.9816514, 546.0550459, 634.1284404, 704.587156,
#      792.6605505, 880.733945, 968.8073394, 1056.880734, 1127.33945, 1215.412844, 1303.486239, 1426.788991, 1532.477064,
#      1620.550459, 1708.623853, 1796.697248, 1867.155963]
# plt.scatter(x, y, s=20, c="g", linewidths=2, alpha=1, marker='o')
# plt.title('CDF of response time',size=16)
plt.xlabel("Response Time (ms)",size=20)
plt.ylabel("CDF",size=20)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams.update({'font.size': 14})
plt.tick_params(labelsize=14)
plt.legend()
plt.savefig('15A.png', dpi=300, bbox_inches='tight')
# plt.show()
