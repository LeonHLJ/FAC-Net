import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d


def basic_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def random_choose(v_len, num_seg):
    start_ind = np.random.randint(0, v_len - num_seg)
    random_p = np.arange(start_ind, start_ind + num_seg)
    return random_p.astype(int)


def random_perturb(v_len, num_seg):
    random_p = np.arange(num_seg) * v_len / num_seg
    for i in range(num_seg):
        if i < num_seg - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)


def uniform_sampling(v_len, num_seg):
    u_sample = np.arange(num_seg) * v_len / num_seg
    u_sample = np.floor(u_sample)
    return u_sample.astype(int)


# Get weighted T CAM and the score for each segment
def get_wtCAM(t_CAM, attention_Weights, num_seq):
    wtCAM = attention_Weights * basic_sigmoid(t_CAM)
    signal = np.reshape(wtCAM, (num_seq, -1, 1))
    score = np.reshape(attention_Weights * t_CAM, (num_seq, -1, 1))
    return signal, score


# Interpolate empty segments
def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')  # linear/quadratic/cubic
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


# Interpolate the wtCAM signals and threshold
def interpolated_wtCAM(cam, scale):
    final_cam = upgrade_resolution(cam, scale)
    result_zero = np.where(final_cam[:, :, 0] < 0.05)
    final_cam[result_zero] = 0
    return final_cam


def minmax_norm(act_map, min_val=None, max_val=None):
    #import pdb; pdb.set_trace()
    if min_val is None or max_val is None:
        max_val = np_relu(np.max(act_map, axis=0))
        min_val = np_relu(np.min(act_map, axis=0))

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val[None, ...]) / delta[None, ...]

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret

def np_relu(inp):
    return (abs(inp) + inp) / 2

# Get the temporal proposal
# def get_temp_proposal(tList, wtcam, c_pred, scale, v_len, num_seq):
#     t_factor = (16 * v_len) / (scale * num_seq * SAMPLING_FRAMES)  # Factor to convert segment index to actual timestamp
#     temp = []
#     for i in range(len(tList)):
#         c_temp = []
#         temp_list = np.array(tList[i])[0]
#         if temp_list.any():
#             grouped_temp_list = grouping(temp_list)  # Get the connected parts
#             for j in range(len(grouped_temp_list)):
#                 c_score = np.mean(wtcam[grouped_temp_list[j], i, 1])
#                 t_start = grouped_temp_list[j][0] * t_factor
#                 t_end = (grouped_temp_list[j][-1] + 1) * t_factor
#                 c_temp.append([c_pred[i], c_score, t_start, t_end])  # Add the proposal
#         temp.append(c_temp)
#     return temp


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i].decode('utf-8')][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)], axis=0)


def idx2multihot(id_list, num_class):
    return np.sum(np.eye(num_class)[id_list], axis=0)


def random_extract(feat, t_max):
    r = np.random.randint(len(feat) - t_max)
    return feat[r:r + t_max]


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len - np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length):
    if len(feat) > length:
        return random_extract(feat, length)
    else:
        return pad(feat, length)


def write_results_to_eval_file(args, dmap, itr1, itr2):
    file_folder = './ckpt/' + args.dataset_name + '/eval/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = str(itr1)
    string_to_write += ' ' + str(itr2)
    for item in dmap:
        string_to_write += ' ' + '%.2f' % item
    fid.write(string_to_write + '\n')
    fid.close()


def write_results_to_file(args, dmap, cmap, itr):
    file_folder = './ckpt/' + args.dataset_name + '/' + str(args.model_id) + '/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' % item
    string_to_write += ' ' + '%.2f' % cmap
    fid.write(string_to_write + '\n')
    fid.close()

def write_settings_to_file(args):
    file_folder = './ckpt/' + args.dataset_name + '/' + str(args.model_id) + '/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = '#' * 80 + '\n'
    for arg in vars(args):
        string_to_write += str(arg) + ': ' + str(getattr(args, arg)) + '\n'
    string_to_write += '*' * 80 + '\n'
    fid.write(string_to_write)
    fid.close()