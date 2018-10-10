#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

dates = [2015, 2016, 2017, 2018]
students = [15, 50, 110, 200]
assistants = [0, 1, 3, 6]

fig, ax1 = plt.subplots(figsize=(6, 3))
ax1.plot(dates, students, '.-', label='students')
ax1.set_ylabel('number of students')
ax2 = ax1.twinx()
ax2.plot(dates, assistants, 'r.--', label='assistants')
ax2.set_ylabel('number of TAs')
ax2.set_xticks(dates)
ax1.set_xlabel('year')
fig.legend(bbox_to_anchor=(0.9, 0.4))
fig.tight_layout()
fig.savefig('enrollment.pdf')
