import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 18})
fig, ax1 = plt.subplots(figsize=(8, 4.8))

x = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
y1 = [0.438287154, 0.475592748, 0.521327014, 0.572972973, 0.617107943, 0.683840749, 0.74270557, 0.785294118,
      0.823529412, 0.870848708, 0.914634146, 0.927927928, 0.954773869]
y2 = [0.789115646, 0.77324263, 0.74829932, 0.721088435, 0.68707483, 0.662131519, 0.634920635, 0.605442177, 0.571428571,
      0.535147392, 0.510204082, 0.467120181, 0.430839002]
z = [0.563562753, 0.588946459, 0.61452514, 0.638554217, 0.650214592, 0.67281106, 0.684596577, 0.683738796, 0.674698795,
     0.662921348, 0.655021834, 0.621417798, 0.59375]

ax1.plot(x, y1, 'g', linewidth=2, label='Precision')
ax1.set_xlabel('\u03BB', size=22)
ax1.set_ylabel('Precision and Recall Rate', size=22)
ax1.plot(x, y2, 'b', linewidth=2, label='Recall')
ax1.plot(x, z, 'r', linewidth=4, label='F-Measure')
ax1.scatter(30, 0.684596577, c='r', marker='o', s=100)
ax1.legend(loc='upper left',fontsize=14)
plt.savefig('figure_5.png', dpi=300, bbox_inches='tight')
