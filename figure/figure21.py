



import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

plt.figure(figsize=(6, 5))

x_data = ["StatuScale","GBMScaler","Showar","Hyscale"]

y_data = [0.5985912303841795,0.5704565854305796,0.5755929218941331,0.5094089614939409]
# z_data=[238,231,222,202,163,112,59]
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


plt.bar(x_data[0], y_data[0],0.4,hatch='\\')
plt.bar(x_data[1], y_data[1],0.4,hatch='+')
plt.bar(x_data[2], y_data[2],0.4,hatch='/')
plt.bar(x_data[3], y_data[3],0.4,hatch='.')
	# plt.bar(x_data[i], z_data[i], 0.4,color='#ff7f0e')
plt.ylim(0, 0.7)

plt.ylabel("Correlation Factor",size=20)

plt.xticks(size=18)

plt.savefig('21.png', dpi=300,bbox_inches='tight')
# plt.show()
