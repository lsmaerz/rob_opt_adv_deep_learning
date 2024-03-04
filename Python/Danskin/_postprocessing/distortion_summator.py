import numpy as np

DIM_PX = 224
NUM_PX = DIM_PX**2
d = 3 * NUM_PX
sqrt_d = np.sqrt(d)

for mode_orig in ['inf', '2', '1']:
    for mode_targ in ['inf', '2', '1']:
        if mode_orig == '1':
            eps = '651.514'
            special = ''
        if mode_orig == '2':
            eps = '2.209'
            special = ''
        if mode_orig == 'inf':
            eps = '0.011764705882352941'
            special = 'geom_harm/'
        if mode_targ == '1':
            factor = d
        if mode_targ == '2':
            factor = sqrt_d
        if mode_targ == 'inf':
            factor = 1
        total = 0
        for rand in ['True', 'False']:
            for decay in ['harmonic', 'geometric']:
                path = './Danskin/_results/Seed_42/ImageNet/L_' + mode_orig + '/' + special + 'stats/L' + mode_targ + '-Distortions_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode_orig + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
                with open(path, 'r') as f:
                    last_line = f.readlines()[-1]
                    numbers = last_line.split(',')
                    # print(sum([float(num) for num in numbers[3:]]))
                    total = total + sum([float(num) for num in numbers[3:]])
        total = factor * total
        total = total / 20
        print(mode_orig + ' -> ' + mode_targ + ' = ' + str(total))
