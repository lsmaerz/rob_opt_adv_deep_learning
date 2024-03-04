import math
import os

import dill as dill
import numpy
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from six.moves import cPickle as pickle
import numpy as np
from numpy import array
from numpy import vdot
from numpy import arccos
from numpy.linalg import norm
import matplotlib.pyplot as plt
#
# ITER = [1, 2, 3]
#
# res = ['']
# for x in ITER:
#     np.concatenate((res, np.concatenate(([''],[str(x)]))))
#
# def heatmap(data):
#     fig, ax = plt.subplots()
#     im = ax.imshow(data, cmap='coolwarm', interpolation='nearest')
#     fig.colorbar(im)
#     # ax.set_xticklabels(np.concatenate((np.array([0]), np.array(ITER))))
#     ax.set_xticklabels(res)
#     # ax.set_yticklabels(np.concatenate((np.array([0]), np.array(ITER))))
#     ax.set_yticklabels(res)
#     plt.savefig('heatmap.svg')
#     plt.clf()
#     plt.cla()

# data = np.random.rand(4, 4) * 2 - 1
# heatmap(data)
ITER = [-2, -1, 0, 1, 2, 4, 8, 16]
NUM_ATTACKS = len(ITER)

def heatmap(data, path, name, min, max, bar):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=bar, interpolation='nearest', vmin=min, vmax=max)
    fig.colorbar(im)
    axis = np.concatenate((np.array(['clean', 'gauss', 'unif', 'bound']), np.array(ITER[3:])))
    ax.set_xticks(np.arange(NUM_ATTACKS+1))
    ax.set_yticks(np.arange(NUM_ATTACKS+1))
    ax.set_xticklabels(axis.tolist(), rotation=45)
    ax.set_yticklabels(axis.tolist())
    for i in range(NUM_ATTACKS+1):
        for j in range(NUM_ATTACKS+1):
            ax.text(j, i, np.round(data[i, j], 2), ha="center", va="center", color="k")
    plt.xlabel("attack")
    plt.ylabel("attack")
    plt.savefig('./' + path + '/' + name + '.pdf')
    plt.clf()
    plt.cla()


angle = np.zeros(500)
means = np.zeros(8)
medians = np.zeros(8)
k = 0
nans = 0
for iter_low in ['clean', '-2', '-1', '0', '1', '2', '4', '8']:
    path = './Danskin/Angles_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_inf_Eps_0.023529411764705882_IterLow_' + iter_low + '_IterHigh_16_Rand_False_Tar_False_Dec_harmonic.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
        for row in range(500):
            line = lines[row]
            if line.__contains__('nan'):
                nans = nans+1
                continue
            number = float(line[2:])
            angle[row-nans] = number
        means[k] = np.average(angle[:-nans])
        medians[k] = np.percentile(angle[:-nans], 50)
        k = k+1
        nans = 0
print('angle means: ', means)
print('angle medians: ', medians)


sim = np.zeros(500)
means = np.zeros(8)
medians = np.zeros(8)
k = 0
nans = 0
for iter_low in ['clean', '-2', '-1', '0', '1', '2', '4', '8']:
    path = './Danskin/Similarity_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_inf_Eps_0.023529411764705882_IterLow_' + iter_low + '_IterHigh_16_Rand_False_Tar_False_Dec_harmonic.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
        for row in range(500):
            line = lines[row]
            if 'nan' in line:
                nans = nans+1
                continue
            number = float(line[2:])
            sim[row-nans] = number
        means[k] = np.average(sim[:-nans])
        medians[k] = np.percentile(sim[:-nans], 50)
        k = k+1
        nans = 0
print('sim means: ', means)
print('sim medians: ', medians)

sim_medians = np.array([[1.0, 0.756537914, 0.70656988, 0.94749707, 0.158449084, 0.05520134, 0.033358777, 0.019658553, 0.01575858],
[0.756537914, 1.0, 0.817725241, 0.747075528, 0.220325544, 0.054336216, 0.014437585, 0.009205604, 0.02561775],
[0.70656988, 0.817725241, 1.0, 0.704741389, 0.234508559, 0.040327443, 0.012771946, 0.030734976, 0.0219689],
[0.94749707, 0.747075528, 0.704741389, 1.0, 0.174155474, 0.072131068, 0.023160099, 0.018420536, 0.01661138],
[0.158449084, 0.220325544, 0.234508559, 0.174155474, 1.0, 0.187269807, 0.147960216, 0.098361854, 0.12005479],
[0.05520134, 0.054336216, 0.040327443, 0.072131068, 0.187269807, 1.0, 0.34759137, 0.18542324, 0.17571744],
[0.033358777, 0.014437585, 0.012771946, 0.023160099, 0.147960216, 0.34759137, 1.0, 0.389107645, 0.33409804],
[0.019658553, 0.009205604, 0.030734976, 0.018420536, 0.098361854, 0.18542324, 0.389107645, 1.0, 0.45773601],
[0.01575858, 0.02561775, 0.0219689, 0.01661138, 0.12005479, 0.17571744, 0.33409804, 0.45773601, 1.0]])

heatmap(sim_medians, './Danskin', 'Repaired_Heat-Sim-Medians_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_inf_Eps_0.023529411764705882_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_False_Tar_False_Dec_harmonic', 0, 1, 'coolwarm')

angle_medians = np.array([[0.0, 40.840064151, 45.043397868, 18.648678157, 80.883113046, 86.835587493, 88.08832521, 88.87357435, 89.09706252],
[40.840064151, 0.0, 35.142106713, 41.66231275, 77.271840264, 86.885228648, 89.172757767, 89.472550293, 88.53205054],
[45.043397868, 35.142106713, 0.0, 45.191313236, 76.437343488, 87.688780384, 89.268201489, 88.238737265, 88.74117352],
[18.648678157, 41.66231275, 45.191313236, 0.0, 79.970481168, 85.863601671, 88.67289796, 88.944521295, 89.04819402],
[80.883113046, 77.271840264, 76.437343488, 79.970481168, 0.0, 79.20649969, 81.491263365, 84.355153466, 83.10473539],
[86.835587493, 86.885228648, 87.688780384, 85.863601671, 79.20649969, 0.0, 69.659935918, 79.314189438, 79.87958844],
[88.08832521, 89.172757767, 89.268201489, 88.67289796, 81.491263365, 69.659935918, 0.0, 67.10096241, 70.48230037],
[88.87357435, 89.472550293, 88.238737265, 88.944521295, 84.355153466, 79.314189438, 67.10096241, 0.0, 62.75888707],
[89.09706252, 88.53205054, 88.74117352, 89.04819402, 83.10473539, 79.87958844, 70.48230037, 62.75888707, 0.0]])

heatmap(angle_medians, './Danskin', 'Repaired_Heat-Angle-Medians_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_inf_Eps_0.023529411764705882_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_False_Tar_False_Dec_harmonic', 0, 90, 'coolwarm_r')

# ---------------------------

sim_means = np.array([[1.0, 0.707224564, 0.650324171, 0.924623422, 0.146772926, 0.051420014, 0.018834088, 0.017995264, 0.01668256],
[0.707224564, 1.0, 0.781231318, 0.690126285, 0.197559569, 0.047840092, 0.00767182, 0.017953306, 0.03033282],
[0.650324171, 0.781231318, 1.0, 0.6387999, 0.207063113, 0.036321585, 0.017505845, 0.033923781, 0.02359317],
[0.924623422, 0.690126285, 0.6387999, 1.0, 0.147631429, 0.049433075, 0.023162764, 0.021236975, 0.02204425],
[0.146772926, 0.197559569, 0.207063113, 0.147631429, 1.0, 0.165621705, 0.132294787, 0.103503573, 0.13102113],
[0.051420014, 0.047840092, 0.036321585, 0.049433075, 0.165621705, 1.0, 0.308782553, 0.191574725, 0.16089043],
[0.018834088, 0.00767182, 0.017505845, 0.023162764, 0.132294787, 0.308782553, 1.0, 0.354206929, 0.30097082],
[0.017995264, 0.017953306, 0.033923781, 0.021236975, 0.103503573, 0.191574725, 0.354206929, 1.0, 0.38413786],
[0.01668256, 0.03033282, 0.02359317, 0.02204425, 0.13102113, 0.16089043, 0.30097082, 0.38413786, 1.0]])

heatmap(sim_means, './Danskin', 'Repaired_Heat-Sim-Means_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_inf_Eps_0.023529411764705882_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_False_Tar_False_Dec_harmonic', 0, 1, 'coolwarm')

angle_means = np.array([[0.0, 43.009901361, 47.489552721, 20.60977044, 81.207993397, 86.939596984, 88.900314583, 88.939284053, 88.9690689],
[43.009901361, 0.0, 37.093529713, 44.425645671, 78.091983165, 87.146864833, 89.545739927, 88.897149583, 88.20960429],
[47.489552721, 37.093529713, 0.0, 48.440541057, 77.539796688, 87.823854552, 88.935202957, 87.941127385, 88.58741259],
[20.60977044, 44.425645671, 48.440541057, 0.0, 81.201314875, 87.057907377, 88.636307281, 88.744960497, 88.64883177],
[81.207993397, 78.091983165, 77.539796688, 81.201314875, 0.0, 80.008990279, 82.089480571, 83.784734772, 82.14433344],
[86.939596984, 87.146864833, 87.823854552, 87.057907377, 80.008990279, 0.0, 71.322781696, 78.451648856, 80.31086111],
[88.900314583, 89.545739927, 88.935202957, 88.636307281, 82.089480571, 71.322781696, 0.0, 68.137609944, 71.68147143],
[88.939284053, 88.897149583, 87.941127385, 88.744960497, 83.784734772, 78.451648856, 68.137609944, 0.0, 65.9354325],
[88.9690689, 88.20960429, 88.58741259, 88.64883177, 82.14433344, 80.31086111, 71.68147143, 65.9354325, 0.0]])

heatmap(angle_means, './Danskin', 'Repaired_Heat-Angle-Means_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_inf_Eps_0.023529411764705882_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_False_Tar_False_Dec_harmonic', 0, 90, 'coolwarm_r')
