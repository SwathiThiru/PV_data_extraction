"""
This file visualizes the extraction performance between TATR and LT.

Author:
    Name:
        Swathi Thiruvengadam
    Email:
        swathi.thiruvengadam@ise.fraunhofer.de
        swathi.thiru078@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
from labelImg import labelImg

# The PV module/cell test data extracted using LT and TATR
labels = ['Solar cells', 'Solar Module']

precision_LT = [0.989, 0.50]
recall_LT = [0.940, 0.50]
f1_LT = [0.964, 0.50]

precision_TATR = [0.6739, 0.5529]
recall_TATR = [0.6739, 0.5529]
f1_TATR = [0.6739, 0.5529]

# Graph 1: Grouped Bar Chart for Precision, Recall, and F1 Score
fig, ax = plt.subplots()
bar_width = 0.2
index = np.arange(len(labels))

bar1 = ax.bar(index - bar_width, precision_LT, bar_width, label='Precision (LT)')
bar2 = ax.bar(index, recall_LT, bar_width, label='Recall (LT)')
bar3 = ax.bar(index + bar_width, f1_LT, bar_width, label='F1 Score (LT)')

bar4 = ax.bar(index + 2*bar_width, precision_TATR, bar_width, label='Precision (TATR)')
bar5 = ax.bar(index + 3*bar_width, recall_TATR, bar_width, label='Recall (TATR)')
bar6 = ax.bar(index + 4*bar_width, f1_TATR, bar_width, label='F1 Score (TATR)')

ax.set_xlabel('Datasheets')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1 Score Comparison')
ax.set_xticks(index + 1.5*bar_width)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Graph 2: Grouped Bar Chart for Precision, Recall, and F1 Score
fig, ax = plt.subplots()
bar_width = 0.35

bar1 = ax.bar(index - bar_width/2, precision_LT, bar_width, label='Precision (LT)')
bar2 = ax.bar(index - bar_width/2, recall_LT, bar_width, label='Recall (LT)')
bar3 = ax.bar(index - bar_width/2, f1_LT, bar_width, label='F1 Score (LT)')

bar4 = ax.bar(index + bar_width/2, precision_TATR, bar_width, label='Precision (TATR)')
bar5 = ax.bar(index + bar_width/2, recall_TATR, bar_width, label='Recall (TATR)')
bar6 = ax.bar(index + bar_width/2, f1_TATR, bar_width, label='F1 Score (TATR)')

ax.set_xlabel('Datasheets')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1 Score Comparison')
ax.set_xticks(index)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Graph 3: Precision-Recall Curve
from sklearn.metrics import precision_recall_curve

precision_LT, recall_LT, _ = precision_recall_curve([1, 0], [1, 0], pos_label=1)
precision_TATR, recall_TATR, _ = precision_recall_curve([1, 0], [1, 0], pos_label=1)

fig, ax = plt.subplots()
ax.plot(recall_LT, precision_LT, label='LT')
ax.plot(recall_TATR, precision_TATR, label='TATR')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
plt.show()

# Graph 4: F1 Score Trend Line
fig, ax = plt.subplots()
ax.plot(labels, f1_LT, label='LT', marker='o')
ax.plot(labels, f1_TATR, label='TATR', marker='o')

ax.set_xlabel('Datasheets')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Trend Line')
ax.legend()
plt.show()

# Graph 5: Confusion Matrix Heatmap
conf_matrix_LT = np.array([[0.989, 0.011], [0.06, 0.50]])
conf_matrix_TATR = np.array([[0.6739, 0.3261], [0.4471, 0.5529]])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

im1 = ax[0].imshow(conf_matrix_LT, cmap='Blues', vmin=0, vmax=1)
ax[0].set_title('LT Confusion Matrix')
ax[0].set_xticks([0, 1])
ax[0].set_yticks([0, 1])
ax[0].set_xticklabels(['Predicted 0', 'Predicted 1'])
ax[0].set_yticklabels(['Actual 0', 'Actual 1'])
plt.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(conf_matrix_TATR, cmap='Blues', vmin=0, vmax=1)
ax[1].set_title('TATR Confusion Matrix')
ax[1].set_xticks([0, 1])
ax[1].set_yticks([0, 1])
ax[1].set_xticklabels(['Predicted 0', 'Predicted 1'])
ax[1].set_yticklabels(['Actual 0', 'Actual 1'])
plt.colorbar(im2, ax=ax[1])

plt.show()
