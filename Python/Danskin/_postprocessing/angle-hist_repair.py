import numpy as np
import matplotlib.pyplot as plt

ITER = [-2, -1, 0, 1, 2, 4, 8, 16]
NUM_ATTACKS = len(ITER)
ANGLE_NUM_BINS = 37
SAMPLE_SIZE = 500

def plot_angle_hist(angle_mean, angle_median, angles, nans):
    plt.figure(figsize=(8, 5))
    plt.hist(np.squeeze(angles[:-nans]), bins=np.linspace(0, 180, ANGLE_NUM_BINS),
             weights=np.ones(SAMPLE_SIZE-nans) / SAMPLE_SIZE, color='gray')
    plt.xticks(np.arange(0, 181, step=int(2 * 180 / (ANGLE_NUM_BINS - 1))))
    plt.axvline(angle_mean, color='orange', linestyle='dotted',
                label='mean = ' + str(round(angle_mean, 2)) + '°')
    plt.axvline(angle_median, color='red', linestyle='dashed',
                label='median = ' + str(round(angle_median, 2)) + '°')
    plt.axvline(90, color='k', linewidth=1, label='orthogonal = 90°')
    plt.legend(loc='upper right')
    plt.xlabel("angle in degrees")
    plt.ylabel("probability")
    plt.savefig('./Danskin/corrected_and_repaired.pdf')
    plt.clf()
    plt.cla()


angle = np.zeros(500)
k = 0
nans = 0
path = './Danskin/stats/Angles_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_inf_Eps_0.023529411764705882_IterLow_1_IterHigh_16_Rand_False_Tar_False_Dec_harmonic.txt'
with open(path, 'r') as f:
    lines = f.readlines()
    for row in range(500):
        line = lines[row]
        if line.__contains__('nan'):
            nans = nans+1
            continue
        number = float(line[2:])
        angle[row-nans] = number
    mean = np.average(angle[:-nans])
    median = np.percentile(angle[:-nans], 50)
    k = k+1
plot_angle_hist(mean, median, angle, nans)

