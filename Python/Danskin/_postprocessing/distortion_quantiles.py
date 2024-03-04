import numpy as np

DIM_PX = 224
NUM_PX = DIM_PX**2
d = 3 * NUM_PX
sqrt_d = np.sqrt(d)
distortions = np.zeros(2500)

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
            decimals = 3
            factor = d
        if mode_targ == '2':
            factor = sqrt_d
            decimals = 5
        if mode_targ == 'inf':
            factor = 1
            decimals = 5
        total = 0
        for rand in ['True', 'False']:
            for decay in ['harmonic', 'geometric']:
                path = './Danskin/_results/Seed_42/ImageNet/L_' + mode_orig + '/' + special + 'stats/L' + mode_targ + '-Distortions_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode_orig + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
                with open(path, 'r') as f:
                    lines = f.readlines()
                    for row in range(500):
                        line = lines[row]
                        numbers = line.split(',')
                        for col in range(5):
                            distortion = float(numbers[3+col])
                            distortions[col * 500 + row] = factor * distortion
        # print('from ' + mode_orig + ' to ' + mode_targ + ' has 99% quantile = ' + str(np.percentile(distortions, 99)))
        # print('from ' + mode_orig + ' to ' + mode_targ + ' has 95% quantile = ' + str(np.percentile(distortions, 95)))
        # print('from ' + mode_orig + ' to ' + mode_targ + ' has 90% quantile = ' + str(np.percentile(distortions, 90)))
        # print('from ' + mode_orig + ' to ' + mode_targ + ' has 50% quantile = ' + str(np.percentile(distortions, 50)))
        # print('from ' + mode_orig + ' to ' + mode_targ + ' has 10% quantile = ' + str(np.percentile(distortions, 10)))
        # print('from ' + mode_orig + ' to ' + mode_targ + ' has 5% quantile = ' + str(np.percentile(distortions, 5)))
        # print('from ' + mode_orig + ' to ' + mode_targ + ' has 1% quantile = ' + str(np.percentile(distortions, 1)))
        print(mode_orig + ' to ' + mode_targ)
        print('[' + str(np.round(np.percentile(distortions, 5), decimals)) + ', ' + str(np.round(np.percentile(distortions, 50), decimals)) + ', ' + str(np.round(np.percentile(distortions, 95), decimals)) + ']')
