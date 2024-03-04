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

MAX_LOSS_IS_KNOWN = True
LOG_OFFSET = 0
SEED = 42
SAMPLE_SIZE = 500
ANGLE_NUM_BINS = 37
SIM_STEP_SIZE = 0.1
DATA_SET = 'ImageNet'
DEFENSE = 'madry'
if DEFENSE in ['natural', 'madry', 'locus']:
    IMAGENET_PATH = './Data/ImageNet/images/'
else:
    IMAGENET_PATH = './Data/ImageNet/imagesRealLabelsMinus1/'
CIFAR10_PATH = './Data/CIFAR-10/cifar-10-batches-py/test_batch'
MADRY_IMAGENET = './Models/Madry/imagenet_linf_8.pt'
# MADRY_IMAGENET = './Models/Madry/imagenet_l2_3_0.pt'
HADI_IMAGENET = './Models/Hadi/imagenet/PGD_1step/imagenet/eps_512/resnet50/noise_1.00/checkpoint.pth.tar' # best model for L2 = 2.250
LOCUS_IMAGENET = './Models/Locus/imagenet_model_weights_2px.pth.tar'
MODE = np.inf
EPS = 6/255
# gaussian, uniform, boundary, iter 1,..., iter 16
ITER = [-2, -1, 0, 1, 2, 4, 8, 16]
NUM_ATTACKS = len(ITER)
RAND = False
TAR = False
torch.manual_seed(SEED)
np.random.seed(SEED)
if DATA_SET == 'ImageNet':
    VAL_SIZE = 50000
    if DEFENSE == 'hadi':
        EXP = [0, 0, 0]
        STD = [1, 1, 1]
    else:
        EXP = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    DIM_PX = 224
elif DATA_SET == 'CIFAR10':
    VAL_SIZE = 10000
    EXP = [0.4914, 0.4822, 0.4465]
    STD = [0.247, 0.243, 0.261]
    DIM_PX = 32
else:
    print('Invalid DATA_SET parameter!')
NUM_PX = DIM_PX**2

# returns cosine similarity between v1 and v2
def cos_sim(v1, v2):
    # return torch.nn.CosineSimilarity(v1, v2)
    # print('v1', v1)
    # print('v1', v2)
    # print('vdot', vdot(v1, v2))
    # print('nv1', norm(v1, 2))
    # print('nv2', norm(v2, 2))
    # print('nv1*nv2', norm(v1, 2) * norm(v2, 2))
    # print('mv1', np.max(v1))
    # print('mv2', np.max(v2))
    return vdot(v1, v2) / (norm(v1, 2) * norm(v2, 2))

# returns radial angle of cosine similarity by taking arccos
def sim_to_angle(sim):
    sim = np.max([-0.999999, np.min([sim, 0.999999])])
    return arccos(sim)

# converts from radial angle to degree angle
def rad_to_deg(r):
    return r * 180 / np.pi

# returns random input tensor of DATA_SET together with ground truth label
def get_random_input_tensor(ds):
    index = int(np.round(np.random.rand() * VAL_SIZE))
    print('Image index = ' + str(index))
    if ds == 'ImageNet':
        path = IMAGENET_PATH
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        label = int(files[index].split('_')[0])
        path = os.path.join(path, files[index])
        img = Image.open(path).convert('RGB')
    elif ds == 'CIFAR10':
        path = CIFAR10_PATH
        f = open(path, 'rb')
        data = pickle.load(f, encoding='bytes')
        f.close()
        img = transforms.ToPILImage()(np.transpose(np.reshape(data[b'data'][index], (3, 32, 32)), (1, 2, 0)))
        label = data[b'labels'][index]
        path = os.path.join(path, str(index))
    else:
        print('Invalid DATA_SET argument')
    print('Image path = ' + str(path))
    return preprocess(img), int(label), index

# normalizes images with mean and standard deviation of training DATA_SET
def preprocess(img):
    if DATA_SET == 'ImageNet':
        pre = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=EXP, std=STD)
        ])
    elif DATA_SET == 'CIFAR10':
        pre = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=EXP, std=STD)
        ])
    else:
        print('Invalid DATA_SET argument')
    input_tensor = pre(img)
    return input_tensor.unsqueeze(0).requires_grad_(True)

# inverts normalization with mean and standard deviation of training DATA_SET
def postprocess(tensor):
    if DATA_SET == 'ImageNet':
        post = transforms.Compose([
            transforms.Normalize(mean=[-EXP[0]/STD[0], -EXP[1]/STD[1], -EXP[2]/STD[2]], std=[1/STD[0], 1/STD[1], 1/STD[2]])
        ])
    elif DATA_SET == 'CIFAR10':
        post = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[-EXP[0]/STD[0], -EXP[1]/STD[1], -EXP[2]/STD[2]], std=[1/STD[0], 1/STD[1], 1/STD[2]])
        ])
    else:
        print('Invalid DATA_SET argument')
    tensor = post(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

# returns (negative) gradient of CEL with respect to input_tensor (for LL-targeted attack)
def get_tensor_gradient(input_tensor, label, targeted):
    output = model(input_tensor)
    confidence = torch.softmax(output, dim=1)
    ll_label = np.argmin(confidence.detach())
    if targeted:
        loss = torch.log(LOG_OFFSET + (1-LOG_OFFSET) * confidence[:, ll_label])
    else:
        loss = -torch.log(LOG_OFFSET + (1-LOG_OFFSET) * confidence[:, int(label)])
    grad = torch.autograd.grad(outputs=loss, inputs=input_tensor)[0]
    return grad, loss

# returns gradient and prediction info of CEL with respect to DNN parameters
def get_model_gradient(input_tensor, label):
    output = model(input_tensor)
    confidence = torch.softmax(output, dim=1)
    ll_label = np.argmin(confidence.detach())
    loss = -torch.log(LOG_OFFSET + (1-LOG_OFFSET) * confidence[:, label])
    grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())[0]
    conf_arr = confidence.detach().numpy()
    top5_pred = np.argsort(-conf_arr[0])[:5]
    top5_conf = conf_arr[0, top5_pred]
    truth_conf = conf_arr[0, label]
    return grad, loss, top5_pred, top5_conf, truth_conf, ll_label

# computes (approximately) uniformly distributed random offset in l_p ball
def uniform_offset(input_tensor, amplitude, mode):
    clone = input_tensor.clone()
    if mode == np.inf:
        delta = amplitude * (2 * torch.rand_like(input_tensor) - 1)
        clone += delta
        clone.unsqueeze(0).requires_grad_(True)
        return clone
    if mode == 2:
        delta = torch.randn_like(input_tensor)
        delta = delta / norm(torch.flatten(delta).detach().numpy(), 2)
    if mode == 1:
        delta = torch.rand_like(input_tensor)
        delta = delta / norm(torch.flatten(delta).detach().numpy(), 1)
    clone += np.random.rand() * delta
    clone.unsqueeze(0).requires_grad_(True)
    return clone

# computes normally distributed random offset
def normal_offset(input_tensor, std):
    delta = std * torch.randn_like(input_tensor)
    clone = input_tensor.clone()
    clone += delta
    clone.unsqueeze(0).requires_grad_(True)
    return clone

# computes random offset onto eps-ball boundary
def boundary_offset(input_tensor, amplitude, mode):
    if mode == np.inf:
        delta = torch.rand_like(input_tensor)
        delta[delta >= 0.5] = amplitude
        delta[delta < 0.5] = -amplitude
    if mode == 2:
        delta = torch.randn_like(input_tensor)
        delta = delta / norm(torch.flatten(delta).detach().numpy(), 2)
    if mode == 1:
        delta = torch.rand_like(input_tensor)
        delta = delta / norm(torch.flatten(delta).detach().numpy(), 1)
    clone = input_tensor.clone()
    clone += delta
    clone.unsqueeze(0).requires_grad_(True)
    return clone

# perturbs input_tensor via input_gradient according to Lp-norm mode, energy, number of iterations and random offsets
def perturb(tensor, label, mode, iter):
    eps = EPS/np.average(STD)
    lower = [-EXP[0]/STD[0], -EXP[1]/STD[1], -EXP[2]/STD[2]]
    upper = [(1-EXP[0])/STD[0], (1 - EXP[1])/STD[1], (1 - EXP[2])/STD[2]]

    # gaussian smoothing
    if iter == -2:
        tensor = normal_offset(tensor, eps / 2)

    # uniform smoothing
    if iter == -1:
        tensor = uniform_offset(tensor, eps, mode)

    # jump to l_p boundary randomly
    if iter == 0:
        tensor = boundary_offset(tensor, eps, mode)

    for k in range(iter):
        if DECAY == 'geometric':
            step = eps * 0.5**k / (2-0.5**(iter-1))
        elif DECAY == 'harmonic':
            step = eps * (1/(k+1)) / sum((1/(x+1)) for x in range(iter))
        else:
            print('Invalid decay style!')
        # offset tensor
        print('    Iter = ' + str(k+1) + ', Step = ' + str(step*EPS/eps))
        if RAND:
            tensor = boundary_offset(tensor, 0.1 * step, MODE)
            step = 0.9 * step
        # compute gradient and its norm in current iterate
        input_gradient, _ = get_tensor_gradient(tensor, label, TAR)
        gradient_norm = norm(torch.flatten(input_gradient).detach().numpy(), mode)
        # update tensor
        if mode == np.inf:
            with torch.no_grad():
                tensor += step * np.sign(input_gradient)
        elif mode == 2:
            with torch.no_grad():
                tensor += (step / gradient_norm) * input_gradient
        elif mode == 1:
            with torch.no_grad():
                tensor += (step / gradient_norm) * input_gradient
        else:
            print('Invalid mode parameter!')
            break

    # projection onto feasible color channel range [0,1] transformed by normalization
    tensor[0, 0, :, :] = torch.clamp(tensor[0, 0, :, :], min=lower[0], max=upper[0])
    tensor[0, 1, :, :] = torch.clamp(tensor[0, 1, :, :], min=lower[1], max=upper[1])
    tensor[0, 2, :, :] = torch.clamp(tensor[0, 2, :, :], min=lower[2], max=upper[2])

    # # projection onto eps-ball
    # origin = tensor.clone()
    # for channel in range(3):
    #     for x in range(DIM_PX):
    #         for y in range(DIM_PX):
    #             tensor[0, channel, x, y] = torch.clamp(tensor[0, channel, x, y],
    #                                                    min=origin[0, channel, x, y].item() - eps,
    #                                                    max=origin[0, channel, x, y].item() + eps)

    return tensor

# returns number of bins for histogramm according to Freedman-Diaconis
def freedman_diaconis(x):
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    return int(bins)

# saves pytorch tensor as pdf image file
def save_tensor_to_pdf(input_tensor, path, name):
    tensor = input_tensor.clone()
    tensor = torch.squeeze(tensor)
    tensor = postprocess(tensor)
    tensor = tensor.detach().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    plt.imshow(tensor)
    plt.show()
    plt.savefig('./' + path + '/' + name + '.pdf')
    plt.clf()
    plt.cla()

# creates heatmap of two-dimensional data
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

# creates angle histogram
def plot_angle_hist(i, j, angle_mean, angle_median, angles):
    plt.figure(figsize=(8, 5))
    plt.hist(np.squeeze(angles[:, i, j]), bins=np.linspace(0, 180, ANGLE_NUM_BINS),
             weights=np.ones(SAMPLE_SIZE) / SAMPLE_SIZE, color='gray')
    plt.xticks(np.arange(0, 181, step=int(2 * 180 / (ANGLE_NUM_BINS - 1))))
    plt.axvline(angle_mean[i][j], color='orange', linestyle='dotted',
                label='mean = ' + str(round(angle_mean[i][j], 2)) + '°')
    plt.axvline(angle_median[i][j], color='red', linestyle='dashed',
                label='median = ' + str(round(angle_median[i][j], 2)) + '°')
    plt.axvline(90, color='k', linewidth=1, label='orthogonal = 90°')
    plt.legend(loc='upper right')
    plt.xlabel("angle in degrees")
    plt.ylabel("probability")
    plt.savefig('./Danskin/histograms/Angle-Hist_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_IterLow_' + ('clean' if i == 0 else str(ITER[i - 1])) + '_IterHigh_' + str(ITER[j - 1]) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
    plt.clf()
    plt.cla()

# creates cosine similarity histogram
def plot_similarity_hist(i, j, similarity_mean, similarity_median, similarity):
    plt.figure(figsize=(8, 5))
    plt.hist(np.squeeze(similarity[:, i, j]), bins=np.linspace(-1, 1, int(2 / SIM_STEP_SIZE) + 1),
             weights=np.ones(SAMPLE_SIZE) / SAMPLE_SIZE, color='gray')
    plt.xticks(np.arange(-1, 1.01, step=2 * SIM_STEP_SIZE))
    plt.axvline(similarity_mean[i][j], color='orange', linestyle='dotted',
                label='mean = ' + str(round(similarity_mean[i][j], 2)))
    plt.axvline(similarity_median[i][j], color='red', linestyle='dashed',
                label='median = ' + str(round(similarity_median[i][j], 2)))
    plt.axvline(0, color='k', linewidth=1, label='orthogonal = 0')
    plt.legend(loc='upper left')
    plt.xlabel("cosine similarity")
    plt.ylabel("probability")
    plt.savefig('./Danskin/histograms/Sim-Hist_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_IterLow_' + ('clean' if i == 0 else str(ITER[i - 1])) + '_IterHigh_' + str(ITER[j - 1]) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
    plt.clf()
    plt.cla()

# creates cumulative distribution function of angles
def plot_angle_cdf(i, j, angle_mean, angle_median, angles):
    counts, bin_edges = np.histogram(angles[:, i, j], bins=np.linspace(0, 180, ANGLE_NUM_BINS))
    cdf = np.cumsum(counts)
    plt.xticks(np.arange(0, 181, step=int(2 * 180 / (ANGLE_NUM_BINS - 1))))
    plt.plot(bin_edges[1:], cdf / cdf[-1], label='CDF', color='navy')
    cdf90 = cdf[int(np.floor(ANGLE_NUM_BINS / 2) - 1)] / SAMPLE_SIZE
    plt.axvline(90, color='k', linewidth=1, label='orthogonal = 90°')
    plt.axhline(cdf90, label='CDF(90°) = ' + str(round(cdf90, 2)), linestyle='dashdot', color='dodgerblue')
    plt.axvline(angle_mean[i][j], color='orange', linestyle='dotted',
                label='mean = ' + str(round(angle_mean[i][j], 2)) + '°')
    plt.axvline(angle_median[i][j], color='red', linestyle='dashed',
                label='median = ' + str(round(angle_median[i][j], 2)) + '°')
    plt.xlabel("angle in degrees")
    plt.ylabel("probability")
    plt.legend(loc='lower right')
    plt.savefig('./Danskin/cdfs/Angle-CDF_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_IterLow_' + ('clean' if i == 0 else str(ITER[i - 1])) + '_IterHigh_' + str(ITER[j - 1]) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
    plt.clf()
    plt.cla()
    return cdf90

# creates cumulative distribution function of cosine similarities
def plot_similarity_cdf(i, j, similarity_mean, similarity_median, similarity, cdf90):
    counts, bin_edges = np.histogram(similarity[:, i, j], bins=np.linspace(-1, 1, int(2 / SIM_STEP_SIZE) + 1))
    cdf = np.cumsum(counts)
    plt.xticks(np.arange(-1, 1.01, step=2 * SIM_STEP_SIZE))
    plt.plot(bin_edges[1:], cdf / cdf[-1], label='CDF', color='navy')
    plt.axvline(0, color='k', linewidth=1, label='orthogonal = 0')
    plt.axhline(1 - cdf90, label='CDF(0) = ' + str(round(1 - cdf90, 2)), linestyle='dashdot', color='dodgerblue')
    plt.axvline(similarity_mean[i][j], color='orange', linestyle='dotted',
                label='mean = ' + str(round(similarity_mean[i][j], 2)))
    plt.axvline(similarity_median[i][j], color='red', linestyle='dashed',
                label='median = ' + str(round(similarity_median[i][j], 2)))
    plt.xlabel("cosine similarity")
    plt.ylabel("probability")
    plt.legend(loc='upper left')
    plt.savefig('./Danskin/cdfs/Sim-CDF_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_IterLow_' + ('clean' if i == 0 else str(ITER[i - 1])) + '_IterHigh_' + str(ITER[j - 1]) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
    plt.clf()
    plt.cla()

# writes one-dimensional data into text file
def write_array_to_txt_file_1d(array, path, name):
    with open('./' + path + '/' + name + '.txt', 'w') as f:
        for num in array:
            f.write(', ' + str(np.round(num, 9)) + '\n')
        f.write('average:\n')
        f.write(str(np.round(np.average(array), 9)))

# writes two-dimensional data into text file
def write_array_to_txt_file_2d(array, path, name):
    with open('./' + path + '/' + name + '.txt', 'w') as f:
        for row in array:
            f.write(', '.join(str(np.round(x, 9)) for x in row) + '\n')
        f.write('averages:\n')
        f.write(', '.join(str(np.round(np.average(array[:, col]), 9)) for col in range(array[0].size)))

# starts all numerical experiments
def compute():
    similarity = np.zeros((SAMPLE_SIZE, NUM_ATTACKS+1, NUM_ATTACKS+1))
    angles = np.zeros((SAMPLE_SIZE, NUM_ATTACKS+1, NUM_ATTACKS+1))
    losses = np.zeros((SAMPLE_SIZE, NUM_ATTACKS+1))
    labels = np.zeros((SAMPLE_SIZE, NUM_ATTACKS+2))
    ll_labels = np.zeros(SAMPLE_SIZE)
    ll_hit = np.zeros((SAMPLE_SIZE, NUM_ATTACKS+1))
    accuracy = np.zeros((SAMPLE_SIZE, NUM_ATTACKS+1))
    top5_acc = np.zeros((SAMPLE_SIZE, NUM_ATTACKS + 1))
    top5_pred = np.zeros((SAMPLE_SIZE, NUM_ATTACKS + 1, 5))
    top5_conf = np.zeros((SAMPLE_SIZE, NUM_ATTACKS + 1, 5))
    pred_confs = np.zeros((SAMPLE_SIZE, NUM_ATTACKS+1))
    truth_confs = np.zeros((SAMPLE_SIZE, NUM_ATTACKS + 1))
    l1_dists = np.zeros((SAMPLE_SIZE, NUM_ATTACKS))
    l2_dists = np.zeros((SAMPLE_SIZE, NUM_ATTACKS))
    linf_dists = np.zeros((SAMPLE_SIZE, NUM_ATTACKS))
    number_of_nan = 0
    number_of_clean_errors = 0

    for k in range(SAMPLE_SIZE):
        print('Sample = ' + str(k+1) + ' from ' + str(SAMPLE_SIZE))
        s = np.average(STD)
        d = 3 * NUM_PX
        sqrt_d = np.sqrt(d)

        # clean image
        recalculate = True
        while recalculate:
            recalculate = False
            clean_tensor, labels[k][0], index = get_random_input_tensor(DATA_SET)
            clean_grad, losses[k][0], top5_pred[k][0], top5_conf[k][0], truth_confs[k][0], ll_labels[k] = get_model_gradient(clean_tensor, int(labels[k][0]))
            if not labels[k][0] == top5_pred[k][0][0]:
                number_of_clean_errors = number_of_clean_errors+1
                recalculate = True
                continue
            labels[k][1] = top5_pred[k][0][0]
            pred_confs[k][0] = top5_conf[k][0][0]
            ll_hit[k][0] = 1 if (ll_labels[k] == labels[k][1]) else 0
            accuracy[k][0] = 1 if (labels[k][0] == labels[k][1]) else 0
            top5_acc[k][0] = 1 if (labels[k][0] in top5_pred[k][0]) else 0
            clean_grad_array = torch.flatten(clean_grad).detach().numpy()
            clean_tensor_array = torch.flatten(clean_tensor).detach().numpy()
            save_tensor_to_pdf(clean_tensor, 'Danskin/images', 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Index_' + str(index) + '_Seed_' + str(
                SEED) + '_K_' + str(SAMPLE_SIZE) + '_Label_' + str(int(labels[k][0])) + '_clean')
            perturbed_tensors = [torch.empty_like(clean_tensor) for _ in range(NUM_ATTACKS)]
            perturbed_tensor_arrays = [np.empty_like(clean_tensor_array) for _ in range(NUM_ATTACKS)]
            perturbed_grads = [torch.empty_like(clean_grad) for _ in range(NUM_ATTACKS)]
            perturbed_grad_arrays = [np.empty_like(clean_grad_array) for _ in range(NUM_ATTACKS)]

            # perturbed images
            for i in range(NUM_ATTACKS):
                print('  Attack = ' + str(i+1) + ' from ' + str(NUM_ATTACKS))
                perturbed_tensors[i] = perturb(tensor=clean_tensor.clone(), label=labels[k][0], mode=MODE, iter=ITER[i])
                perturbed_tensor_arrays[i] = torch.flatten(perturbed_tensors[i]).detach().numpy()
                save_tensor_to_pdf(perturbed_tensors[i], 'Danskin/images', 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Index_' + str(
                    index) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Label_' + str(int(labels[k][0])) + '_Mode_' + str(
                    MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER[i]) + '_Rand_' + str(
                    RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
                perturbed_grads[i], losses[k][i+1], top5_pred[k][i+1], top5_conf[k][i+1], truth_confs[k][i+1], _ = get_model_gradient(perturbed_tensors[i], int(labels[k][0]))
                labels[k][i + 2] = top5_pred[k][i+1][0]
                pred_confs[k][i+1] = top5_conf[k][i+1][0]
                if np.isnan(losses[k][i+1]):
                    number_of_nan = number_of_nan+1
                    recalculate = True
                    print('  SKIP image no. ' + str(index) + ' since loss is NaN!')
                    break
                print('  Loss =', losses[k][i+1])
                perturbed_grad_arrays[i] = torch.flatten(perturbed_grads[i]).detach().numpy()
                ll_hit[k][i+1] = 1 if (ll_labels[k] == labels[k][i+2]) else 0
                accuracy[k][i+1] = 1 if (labels[k][0] == labels[k][i+2]) else 0
                top5_acc[k][i+1] = 1 if (labels[k][0] in top5_pred[k][i+1]) else 0
                l1_dists[k][i] = norm((clean_tensor_array - perturbed_tensor_arrays[i]) * s, 1) / d
                l2_dists[k][i] = norm((clean_tensor_array - perturbed_tensor_arrays[i]) * s, 2) / sqrt_d
                linf_dists[k][i] = norm((clean_tensor_array - perturbed_tensor_arrays[i]) * s, np.inf)
                similarity[k][0][i + 1] = cos_sim(clean_grad_array, perturbed_grad_arrays[i])
                angles[k][0][i + 1] = rad_to_deg(sim_to_angle(similarity[k][0][i + 1]))
                for j in range(i):
                    similarity[k][j + 1][i + 1] = cos_sim(perturbed_grad_arrays[j], perturbed_grad_arrays[i])
                    angles[k][j + 1][i + 1] = rad_to_deg(sim_to_angle(similarity[k][j + 1][i + 1]))

    print('Number of NaN losses = ' + str(number_of_nan))
    print('Number of clean errors = ' + str(number_of_clean_errors))

    # fill lower left triangle matrix symmetrically
    for i in range(SAMPLE_SIZE):
        similarity[i, :, :] = similarity[i, :, :] + np.transpose(similarity[i, :, :])
        angles[i, :, :] = angles[i, :, :] + np.transpose(angles[i, :, :])
    for i in range(NUM_ATTACKS+1):
        similarity[:, i, i] = 1
        angles[:, i, i] = 0

    # log results into text files
    similarity_mean = np.zeros((NUM_ATTACKS + 1, NUM_ATTACKS + 1))
    similarity_median = np.zeros((NUM_ATTACKS + 1, NUM_ATTACKS + 1))
    angle_mean = np.zeros((NUM_ATTACKS + 1, NUM_ATTACKS + 1))
    angle_median = np.zeros((NUM_ATTACKS + 1, NUM_ATTACKS + 1))
    for i in range(NUM_ATTACKS+1):
        write_array_to_txt_file_2d(top5_pred[:, i, :], 'Danskin/stats',
                                   'Top5-Pred_' + 'Dataset_' + DATA_SET + '_Defense_' + str(
                                       DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(
                                       MODE) + '_Eps_' + str(EPS) + '_Iter_' + ('clean' if i == 0 else str(ITER[i-1])) + '_Rand_' + str(
                                       RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
        write_array_to_txt_file_2d(top5_conf[:, i, :], 'Danskin/stats',
                                   'Top5-Conf_' + 'Dataset_' + DATA_SET + '_Defense_' + str(
                                       DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(
                                       MODE) + '_Eps_' + str(EPS) + '_Iter_' + ('clean' if i == 0 else str(ITER[i-1])) + '_Rand_' + str(
                                       RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))

        for j in range(NUM_ATTACKS+1):
            similarity_mean[i][j] = np.mean(similarity[:, i, j])
            similarity_median[i][j] = np.median(similarity[:, i, j])
            angle_mean[i][j] = np.mean(angles[:, i, j])
            angle_median[i][j] = np.median(angles[:, i, j])
            if j > i:
                write_array_to_txt_file_1d(similarity[:, i, j], 'Danskin/stats', 'Similarity_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_IterLow_' + ('clean' if i == 0 else str(ITER[i-1])) + '_IterHigh_' + str(ITER[j-1]) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
                write_array_to_txt_file_1d(angles[:, i, j], 'Danskin/stats', 'Angles_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_IterLow_' + ('clean' if i == 0 else str(ITER[i-1])) + '_IterHigh_' + str(ITER[j-1]) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))

    write_array_to_txt_file_2d(similarity_mean, 'Danskin/stats', 'Similarity-Mean_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(similarity_median, 'Danskin/stats', 'Similarity-Median_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(angle_mean, 'Danskin/stats', 'Angle-Mean_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(angle_median, 'Danskin/stats', 'Angle-Median_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(losses, 'Danskin/stats', 'Losses_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(labels, 'Danskin/stats', 'Labels_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(accuracy, 'Danskin/stats', 'Accuracy_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(top5_acc, 'Danskin/stats', 'Top5-Accuracy_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(pred_confs, 'Danskin/stats', 'Prediction-Confidences_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(truth_confs, 'Danskin/stats', 'True-Label-Confidences_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(l1_dists, 'Danskin/stats', 'L1-Distortions_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(l2_dists, 'Danskin/stats', 'L2-Distortions_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(linf_dists, 'Danskin/stats', 'Linf-Distortions_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_2d(ll_hit, 'Danskin/stats', 'LL-Hits_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))
    write_array_to_txt_file_1d(ll_labels, 'Danskin/stats', 'LL-Labels_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY))

    # plot heatmaps
    heatmap(similarity_mean, 'Danskin/heatmaps', 'Heat-Sim-Means' + '_Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf', 0, 1, 'coolwarm')
    heatmap(similarity_median, 'Danskin/heatmaps', 'Heat-Sim-Medians' + '_Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf', 0, 1, 'coolwarm')
    heatmap(angle_mean, 'Danskin/heatmaps', 'Heat-Angle-Means' + '_Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf', 0, 90, 'coolwarm_r')
    heatmap(angle_median, 'Danskin/heatmaps', 'Heat-Angle-Medians' + '_Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf', 0, 90, 'coolwarm_r')

    # plot CEL histogram for all attacks
    max_loss = int(np.ceil(np.nanmax(losses)))
    stepsize = np.round(max_loss/10, 2)
    losses_without_nan = np.zeros((SAMPLE_SIZE, NUM_ATTACKS + 1))
    overflow = False
    for i in range(SAMPLE_SIZE):
        for j in range(NUM_ATTACKS+1):
            if np.isnan(losses[i, j]):
                losses_without_nan[i, j] = max_loss+stepsize
                overflow = True
            else:
                losses_without_nan[i, j] = losses[i, j]

    if overflow:
        plt.hist(losses_without_nan, bins=np.append(np.linspace(0, max_loss, 11), [max_loss + stepsize]),
                 weights=np.ones((SAMPLE_SIZE, NUM_ATTACKS + 1)) / SAMPLE_SIZE, label=np.concatenate(
                (np.array(['clean', 'gauss', 'unif', 'bound']), np.array(['iter=' + str(x) + ', rand=' + str(RAND) for x in ITER[3:]]))))
        plt.xticks(np.append(np.arange(0, max_loss + 1, step=stepsize), [max_loss + stepsize]))
        plt.xlabel("CEL (with " + str(max_loss+stepsize) + " as +infinity)")
    else:
        plt.hist(losses, bins=np.linspace(0, max_loss, 11),
                 weights=np.ones((SAMPLE_SIZE, NUM_ATTACKS + 1)) / SAMPLE_SIZE, label=np.concatenate(
                (np.array(['clean', 'gauss', 'unif', 'bound']), np.array(['iter=' + str(x) + ', rand=' + str(RAND) for x in ITER[3:]]))))
        plt.xticks(np.arange(0, max_loss + 1, step=stepsize))
        plt.xlabel("CEL")
    plt.ylabel("probability")
    plt.legend(loc='upper right')
    plt.savefig('./Danskin/histograms/CEL-Hist_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
    plt.clf()
    plt.cla()

    if MAX_LOSS_IS_KNOWN:
        for KNOWN_MAX_LOSS in [70, 80, 90, 100, 110, 120, 130]:
            stepsize = np.round(KNOWN_MAX_LOSS / 10, 2)
            plt.hist(losses, bins=np.linspace(0, KNOWN_MAX_LOSS, 11),
                     weights=np.ones((SAMPLE_SIZE, NUM_ATTACKS + 1)) / SAMPLE_SIZE, label=np.concatenate(
                    (np.array(['clean', 'gauss', 'unif', 'bound']), np.array(['iter=' + str(x) + ', rand=' + str(RAND) for x in ITER[3:]]))))
            plt.xticks(np.arange(0, KNOWN_MAX_LOSS + 1, step=stepsize))
            plt.xlabel("CEL")
            plt.ylabel("probability")
            plt.legend(loc='upper right')
            plt.savefig(
                './Danskin/histograms/CEL-Hist_' + 'Known-Max-Loss_' + str(KNOWN_MAX_LOSS) + '_Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(
                    SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(
                    ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
            plt.clf()
            plt.cla()

    # plot CEL histogram for gradient-free attacks
    max_loss = int(np.ceil(np.nanmax(losses[:, :4])))
    stepsize = np.round(max_loss / 10, 2)
    losses_without_nan = np.zeros((SAMPLE_SIZE, 4))
    overflow = False
    for i in range(SAMPLE_SIZE):
        for j in range(4):
            if np.isnan(losses[i, j]):
                losses_without_nan[i, j] = max_loss + stepsize
                overflow = True
            else:
                losses_without_nan[i, j] = losses[i, j]

    if overflow:
        plt.hist(losses_without_nan, bins=np.append(np.linspace(0, max_loss, 11), [max_loss + stepsize]),
                 weights=np.ones((SAMPLE_SIZE, 4)) / SAMPLE_SIZE, label=np.array(['clean', 'gauss', 'unif', 'bound']))
        plt.xticks(np.append(np.arange(0, max_loss + 0.1, step=stepsize), [max_loss + stepsize]))
        plt.xlabel("CEL (with " + str(max_loss + stepsize) + " as +infinity)")
    else:
        plt.hist(losses[:, :4], bins=np.linspace(0, max_loss, 11),
                 weights=np.ones((SAMPLE_SIZE, 4)) / SAMPLE_SIZE, label=np.array(['clean', 'gauss', 'unif', 'bound']))
        plt.xticks(np.arange(0, max_loss + 0.1, step=stepsize))
        plt.xlabel("CEL")
    plt.ylabel("probability")
    plt.legend(loc='upper right')
    plt.savefig(
        './Danskin/histograms/CEL-Hist-Only-Gradient-Free_' + 'Dataset_' + DATA_SET + '_Defense_' + str(DEFENSE) + '_Seed_' + str(
            SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(
            ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
    plt.clf()
    plt.cla()

    if MAX_LOSS_IS_KNOWN:
        for KNOWN_MAX_LOSS in [1, 5, 10, 20, 40, 80, 120]:
            stepsize = np.round(KNOWN_MAX_LOSS / 10, 2)
            plt.hist(losses[:, :4], bins=np.linspace(0, KNOWN_MAX_LOSS, 11),
                     weights=np.ones((SAMPLE_SIZE, 4)) / SAMPLE_SIZE,
                     label=np.array(['clean', 'gauss', 'unif', 'bound']))
            plt.xticks(np.arange(0, KNOWN_MAX_LOSS + 0.1, step=stepsize))
            plt.ylabel("probability")
            plt.legend(loc='upper right')
            plt.savefig(
                './Danskin/histograms/CEL-Hist-Only-Gradient-Free_' + 'Known-Max-Loss_' + str(KNOWN_MAX_LOSS) + '_Dataset_' + DATA_SET + '_Defense_' + str(
                    DEFENSE) + '_Seed_' + str(
                    SEED) + '_K_' + str(SAMPLE_SIZE) + '_Mode_' + str(MODE) + '_Eps_' + str(EPS) + '_Iter_' + str(
                    ITER) + '_Rand_' + str(RAND) + '_Tar_' + str(TAR) + '_Dec_' + str(DECAY) + '.pdf')
            plt.clf()
            plt.cla()

    # plot histograms and cdfs for angles and cosine similarities
    for i in range(NUM_ATTACKS+1):
        for j in range(i+1, NUM_ATTACKS+1):
            plot_angle_hist(i, j, angle_mean, angle_median, angles)
            cdf90 = plot_angle_cdf(i, j, angle_mean, angle_median, angles)
            plot_similarity_hist(i, j, similarity_mean, similarity_median, similarity)
            plot_similarity_cdf(i, j, similarity_mean, similarity_median, similarity, cdf90)

# load the pretrained ResNet50 model
# if torch.cuda.is_available():
#     checkpoint = torch.load(resume_path, pickle_module=dill)
# else:
#     checkpoint = torch.load(resume_path, pickle_module=dill, map_location=torch.device('cpu'))

if DATA_SET == 'ImageNet':
    if DEFENSE == 'natural':
        # works
        model = models.resnet50(pretrained=True)
    elif DEFENSE == 'madry':
        # works
        state_dict_path = 'model'
        model = models.resnet50(num_classes=1000)
        checkpoint = torch.load(MADRY_IMAGENET)
        sd = {}
        for k in checkpoint[state_dict_path]:
            print('old name =', k)
            k_without_module = k[len('module.'):]
            v = checkpoint[state_dict_path][k]
            if k.startswith('module.'):
                sd[k_without_module] = v
            else:
                sd[k] = v
            print('new name =', k_without_module)
        sd_new = {}
        for k in sd:
            if k.startswith('attacker.') or k in ["normalizer.new_mean", "normalizer.new_std"]:
                continue
            k_without_model = k[len('model.'):]
            v = sd[k]
            if k.startswith('model.'):
                sd_new[k_without_model] = v
            else:
                sd_new[k] = v
        model.load_state_dict(sd_new)
    elif DEFENSE == 'hadi':
        # works
        model = models.resnet50(num_classes=1000)
        checkpoint = torch.load(HADI_IMAGENET)
        state_dict_path = 'state_dict'
        sd = {}
        for k in checkpoint[state_dict_path]:
            k_without_prefix = k[len('1.module.'):]
            v = checkpoint[state_dict_path][k]
            if k.startswith('1.module.'):
                sd[k_without_prefix] = v
            else:
                sd[k] = v
        model.load_state_dict(sd)
    elif DEFENSE == 'locus':
        model = models.resnet50(num_classes=1000)
        checkpoint = torch.load(LOCUS_IMAGENET)
        state_dict_path = 'state_dict'
        sd = {}
        for k in checkpoint[state_dict_path]:
            k_without_module = k[len('module.'):]
            v = checkpoint[state_dict_path][k]
            if k.startswith('module.'):
                sd[k_without_module] = v
            else:
                sd[k] = v
        model.load_state_dict(sd)
elif DATA_SET == 'CIFAR10':
    if DEFENSE in ['natural', 'madry']:
        model = models.resnet50(num_classes=10)
        checkpoint = torch.load('./Models/Madry/cifar_nat.pt')
        state_dict_path = 'model'
        sd = {}
        for k in checkpoint[state_dict_path]:
            print('old name =', k)
            k_without_module = k[len('module.'):]
            v = checkpoint[state_dict_path][k]
            if k.startswith('module.'):
                sd[k_without_module] = v
            else:
                sd[k] = v
            print('new name =', k_without_module)
        sd_new = {}
        for k in sd:
            if k.startswith('attacker.') or k in ["normalizer.new_mean", "normalizer.new_std"]:
                continue
            k_without_model = k[len('model.'):]
            v = sd[k]
            if k.startswith('model.'):
                sd_new[k_without_model] = v
            else:
                sd_new[k] = v
        model.load_state_dict(sd_new)
    if DEFENSE == 'hadi':
        models.resnet110(num_classes=10)
        checkpoint = torch.load('./Models/Hadi/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_10steps/weight_1.0/eps_64/cifar10/resnet110/noise_0.12/checkpoint.pth.tar')
        model.load_state_dict(checkpoint)
else:
    print('Invalid DATA_SET parameter!')

# set the model to evaluation mode
model.eval()

# start computation

# seed 42
SEED = 42

DECAY = 'harmonic'
RAND = False
compute()
print('------------------------------------Harmonic & Det = DONE-------------------------------------')

DECAY = 'harmonic'
RAND = True
compute()
print('------------------------------------Harmonic & Rand = DONE-------------------------------------')

DECAY = 'geometric'
RAND = False
compute()
print('------------------------------------Geometric & Det = DONE-------------------------------------')

DECAY = 'geometric'
RAND = True
compute()
print('------------------------------------Geometric & Rand = DONE-------------------------------------')

print('------------------------------------SEED = ' + str(SEED) + 'DONE--------------------------------')

# seed 0
SEED = 0

DECAY = 'harmonic'
RAND = False
compute()
print('------------------------------------Harmonic & Det = DONE-------------------------------------')

DECAY = 'harmonic'
RAND = True
compute()
print('------------------------------------Harmonic & Rand = DONE-------------------------------------')

DECAY = 'geometric'
RAND = False
compute()
print('------------------------------------Geometric & Det = DONE-------------------------------------')

DECAY = 'geometric'
RAND = True
compute()
print('------------------------------------Geometric & Rand = DONE-------------------------------------')

print('------------------------------------SEED = ' + str(SEED) + 'DONE--------------------------------')
