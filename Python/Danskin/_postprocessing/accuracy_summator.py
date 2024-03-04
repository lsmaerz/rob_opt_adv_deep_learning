import numpy as np
print('Vol TOP-1:')
for decay in ['harmonic', 'geometric']:
    for rand in ['False', 'True']:
        row = decay[:4] + '. & ' + rand + ' & '
        for mode in ['inf', '2', '1']:
            eps = ''
            if mode == '1':
                eps = '651.514'
            if mode == '2':
                eps = '2.209'
            if mode == 'inf':
                eps = '0.011764705882352941'
            path = './Danskin/_results/Seed_42/ImageNet/L_' + mode + '/stats/Accuracy_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
            with open(path, 'r') as f:
                last_line = f.readlines()[-1]
                numbers = last_line.split(',')
                total = sum([float(num) for num in numbers[2:]])
                total = np.round(100*total/len(numbers[2:]), 2)
            row = row + str(total) + ' & '
        print(row[:-2] + ' \\\\')

print('Vol TOP-5:')
for decay in ['harmonic', 'geometric']:
    for rand in ['False', 'True']:
        row = decay[:4] + '. & ' + rand + ' & '
        for mode in ['inf', '2', '1']:
            eps = ''
            if mode == '1':
                eps = '651.514'
            if mode == '2':
                eps = '2.209'
            if mode == 'inf':
                eps = '0.011764705882352941'
            path = './Danskin/_results/Seed_42/ImageNet/L_' + mode + '/stats/Top5-Accuracy_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
            with open(path, 'r') as f:
                last_line = f.readlines()[-1]
                numbers = last_line.split(',')
                total = sum([float(num) for num in numbers[2:]])
                total = np.round(100*total/len(numbers[2:]), 2)
            row = row + str(total) + ' & '
        print(row[:-2] + ' \\\\')

print('Nest TOP-1:')
for decay in ['harmonic', 'geometric']:
    for rand in ['False', 'True']:
        row = decay[:4] + '. & ' + rand + ' & '
        for mode in ['inf', '2', '1']:
            eps = ''
            if mode == '1':
                eps = '1770.918'
            if mode == '2':
                eps = '4.564'
            if mode == 'inf':
                eps = '0.011764705882352941'
            path = './Danskin/_results/Seed_42/ImageNet/L_' + mode + '/stats/Accuracy_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
            with open(path, 'r') as f:
                last_line = f.readlines()[-1]
                numbers = last_line.split(',')
                total = sum([float(num) for num in numbers[2:]])
                total = np.round(100*total/len(numbers[2:]), 2)
            row = row + str(total) + ' & '
        print(row[:-2] + ' \\\\')

print('Nest TOP-5:')
for decay in ['harmonic', 'geometric']:
    for rand in ['False', 'True']:
        row = decay[:4] + '. & ' + rand + ' & '
        for mode in ['inf', '2', '1']:
            eps = ''
            if mode == '1':
                eps = '1770.918'
            if mode == '2':
                eps = '4.564'
            if mode == 'inf':
                eps = '0.011764705882352941'
            path = './Danskin/_results/Seed_42/ImageNet/L_' + mode + '/stats/Top5-Accuracy_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
            with open(path, 'r') as f:
                last_line = f.readlines()[-1]
                numbers = last_line.split(',')
                total = sum([float(num) for num in numbers[2:]])
                total = np.round(100*total/len(numbers[2:]), 2)
            row = row + str(total) + ' & '
        print(row[:-2] + ' \\\\')

print('Vol Loss:')
for decay in ['harmonic', 'geometric']:
    for rand in ['False', 'True']:
        row = decay[:4] + '. & ' + rand + ' & '
        for mode in ['inf', '2', '1']:
            eps = ''
            if mode == '1':
                eps = '651.514'
            if mode == '2':
                eps = '2.209'
            if mode == 'inf':
                eps = '0.011764705882352941'
            path = './Danskin/_results/Seed_42/ImageNet/L_' + mode + '/stats/Losses_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
            with open(path, 'r') as f:
                last_line = f.readlines()[-1]
                numbers = last_line.split(',')
                total = sum([float(num) for num in numbers[2:]])
                total = np.round(100*total/len(numbers[2:]), 2)
            row = row + str(total) + ' & '
        print(row[:-2] + ' \\\\')

print('Nest Loss:')
for decay in ['harmonic', 'geometric']:
    for rand in ['False', 'True']:
        row = decay[:4] + '. & ' + rand + ' & '
        for mode in ['inf', '2', '1']:
            eps = ''
            if mode == '1':
                eps = '1770.918'
            if mode == '2':
                eps = '4.564'
            if mode == 'inf':
                eps = '0.011764705882352941'
            path = './Danskin/_results/Seed_42/ImageNet/L_' + mode + '/stats/Losses_Dataset_ImageNet_Defense_natural_Seed_42_K_500_Mode_' + mode + '_Eps_' + eps + '_Iter_[-2, -1, 0, 1, 2, 4, 8, 16]_Rand_' + rand + '_Tar_False_Dec_' + decay + '.txt'
            with open(path, 'r') as f:
                last_line = f.readlines()[-1]
                numbers = last_line.split(',')
                total = sum([float(num) for num in numbers[2:]])
                total = np.round(100*total/len(numbers[2:]), 2)
            row = row + str(total) + ' & '
        print(row[:-2] + ' \\\\')