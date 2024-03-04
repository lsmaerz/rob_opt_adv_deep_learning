import numpy as np

DIM_PX = 224
NUM_PX = DIM_PX**2
d = 3 * NUM_PX
sqrt_d = np.sqrt(d)
losses = np.zeros(2500)

for mode_orig in ['inf', '2', '1']:
    if mode_orig == '1':
        eps = '651.514'
        eps = '1770.918'
        special = ''
        decimals = 2
    if mode_orig == '2':
        eps = '2.209'
        eps = '4.564'
        special = ''
        decimals = 2
    if mode_orig == 'inf':
        eps = '0.011764705882352941'
        special = 'geom_harm/'
        decimals = 2
    for rand in ['True', 'False']:
        for decay in ['harmonic', 'geometric']:
            path = './Danskin/_results/Seed_42/ImageNet/L_' + mode_orig + '/' + special + 'stats/Losses_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode_orig + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
            with open(path, 'r') as f:
                lines = f.readlines()
                for row in range(500):
                    line = lines[row]
                    numbers = line.split(',')
                    for col in range(5):
                        loss = float(numbers[4+col])
                        losses[col * 500 + row] = loss
    print(mode_orig)
    print('[' + str(np.round(np.percentile(losses, 5), decimals)) + ', ' + str(
        np.round(np.percentile(losses, 50), decimals)) + ', ' + str(
        np.round(np.percentile(losses, 95), decimals)) + ']')
