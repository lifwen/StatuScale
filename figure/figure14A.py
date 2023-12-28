import matplotlib.pyplot as plt
import numpy

plt.rc('font',family='Times New Roman')
similarity = [53.211 ,58.747,60.173,62.862]
divergence = [270.540 ,310.660,281.600,316.620 ]
labels = ['StatuScale',"GBMScaler",'Showar','Hyscale']

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


width = 0.25
x1_list = []
x2_list = []
for i in range(len(similarity)):
    x1_list.append(i)
    x2_list.append(i + width)

fig, ax1 = plt.subplots(figsize=(6,5))

ax1.set_ylabel('Average Response Time (ms)',size=20)
ax1.set_ylim(0, 80)
ax1.bar(x1_list, similarity, color='darkorange',width=width, align='edge',label='Truly UnderLoad', hatch='/')


ax2 = ax1.twinx()
ax2.set_ylabel('99th Percentile Response Time (ms)',size=20)
ax2.set_xlabel('1234',size=14)
ax2.set_ylim(0, 400)

ax2.bar(x2_list, divergence, width=width, align='edge',label='99th Percentile Response Time',hatch='\\')

ax2.bar([0], [0], width=width, align='edge',label='Average Response Time', hatch='/')
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.legend(loc="upper right",fontsize='medium')
x = numpy.arange(0.25,4.25,1)
plt.xticks(x, labels,size=20)

plt.savefig('14A.png', dpi=300, bbox_inches='tight')
# plt.show()