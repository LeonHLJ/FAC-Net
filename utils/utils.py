import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i].decode('utf-8')][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)], axis=0)


def idx2multihot(id_list, num_class):
    return np.sum(np.eye(num_class)[id_list], axis=0)


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
