

#
import matplotlib.pyplot as plt

import numpy
plt.rc('font',family='Times New Roman')
similarity = [3.340875591832304,4.9626176571079705,3.746336808663564,5.196141300564627]  # similarity of action
divergence = [1.371988303405547,2.4676939526969366,1.5870847734566262,2.736440045338262]  # js diversity
labels = ['StatuScale','GBMScaler', 'Showar',"Hyscale"]

plt.rcParams['axes.labelsize'] = 14  # xy轴label的size
plt.rcParams['xtick.labelsize'] = 14  # x轴ticks的size
plt.rcParams['ytick.labelsize'] = 14  # y轴ticks的size
plt.rcParams['legend.fontsize'] = 12  # 图例的size


width = 0.25
x1_list = []
x2_list = []
for i in range(len(similarity)):
    x1_list.append(i)
    x2_list.append(i + width)

fig, ax1 = plt.subplots(figsize=(6,5))

ax1.set_ylabel('SLO Violation Rate (%)',size=20)

ax1.set_ylim(0, 7)
ax1.bar(x1_list, similarity, width=width, color='darkorange', align='edge',label='SLO=200 ms',hatch='/')


ax1.bar(x2_list, divergence, width=width, align='edge',label='SLO=250 ms',hatch='\\')
# plt.xlabel('entry a')
# ax1.bar([0], [0], width=width, color='lightseagreen', align='edge',label='XGBoost')
plt.tick_params(labelsize=14)
# plt.title('Comparison of SLO violation rates using different methods', size=14)
plt.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.legend(loc="upper right",fontsize='medium')
x = numpy.arange(0.25,4.25,1)
# 用第1组...替换横坐标x的值
plt.xticks(x, labels,size=20)
# X1=[0.979*4800,0.951*4800,0.914*4800,0.831*4800,0.671*4800,0.461*4800,0.243*4800]
# Z1=[0.979,0.951,0.914,0.831,0.671,0.461,0.243]
# for x, y,z in zip(x2_list, X1,Z1):
#     plt.text(x+0.15, y+30, z, ha='center', va='bottom', fontsize=10.5)
# plt.xlabel('entry a')
# plt.plot(x2_list, X1, label="X1", color="#FF3B1D", marker='*', linestyle="-")
plt.savefig('16A.png', dpi=300, bbox_inches='tight')
# plt.show()