import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils.utils as utils
from .classificationMAP import getClassificationMAP as cmAP
from .detectionMAP import getSingleStreamDetectionMAP as dsmAP
from .detectionMAP import getTwoStreamDetectionMAP as dtmAP


def ss_eval(epoch, dataloader, args, logger, model, device):
    vid_preds = []
    frm_preds = []
    vid_lens = []
    labels = []

    for num, sample in enumerate(dataloader):
        if (num + 1) % 100 == 0:
            print('Testing test data point %d of %d' % (num + 1, len(dataloader)))

        features = sample['data'].numpy()
        label = sample['labels'].numpy()
        vid_len = sample['vid_len'].numpy()

        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
            _, vid_pred, _, frm_scr = model(Variable(features))
            frm_pred = F.softmax(frm_scr, -1)
            vid_pred = np.squeeze(vid_pred.cpu().data.numpy(), axis=0)
            frm_pred = np.squeeze(frm_pred.cpu().data.numpy(), axis=0)
            label = np.squeeze(label, axis=0)

        vid_preds.append(vid_pred)
        frm_preds.append(frm_pred)
        vid_lens.append(vid_len)
        labels.append(label)

    vid_preds = np.array(vid_preds)
    frm_preds = np.array(frm_preds)
    vid_lens = np.array(vid_lens)
    labels = np.array(labels)

    cmap = cmAP(vid_preds, labels)
    dmap, iou = dsmAP(vid_preds, frm_preds, vid_lens, dataloader.dataset.path_to_annotations, args)

    print('Classification map %f' % cmap)
    for item in list(zip(iou, dmap)):
        print('Detection map @ %f = %f' % (item[0], item[1]))

    logger.log_value('Test Classification mAP', cmap, epoch)
    for item in list(zip(dmap, iou)):
        logger.log_value('Test Detection1 mAP @ IoU = ' + str(item[1]), item[0], epoch)

    utils.write_results_to_file(args, dmap, cmap, epoch)


def ts_eval(dataloader, args, logger, rgb_model, flow_model, device):
    rgb_vid_preds = []
    rgb_frame_preds = []
    flow_vid_preds = []
    flow_frame_preds = []
    vid_lens = []
    labels = []

    for num, sample in enumerate(dataloader):
        if (num + 1) % 100 == 0:
            print('Testing test data point %d of %d' % (num + 1, len(dataloader)))

        rgb_features = sample['rgb_data'].numpy()
        flow_features = sample['flow_data'].numpy()
        label = sample['labels'].numpy()
        vid_len = sample['vid_len'].numpy()

        rgb_features_inp = torch.from_numpy(rgb_features).float().to(device)
        flow_features_inp = torch.from_numpy(flow_features).float().to(device)

        with torch.no_grad():
            _, rgb_video_pred, _, rgb_frame_scr = rgb_model(Variable(rgb_features_inp))
            _, flow_video_pred, _, flow_frame_scr = flow_model(Variable(flow_features_inp))

            rgb_frame_pred = F.softmax(rgb_frame_scr, -1)
            flow_frame_pred = F.softmax(flow_frame_scr, -1)

            rgb_frame_pred = np.squeeze(rgb_frame_pred.cpu().data.numpy(), axis=0)
            flow_frame_pred = np.squeeze(flow_frame_pred.cpu().data.numpy(), axis=0)
            rgb_video_pred = np.squeeze(rgb_video_pred.cpu().data.numpy(), axis=0)
            flow_video_pred = np.squeeze(flow_video_pred.cpu().data.numpy(), axis=0)
            label = np.squeeze(label, axis=0)

        rgb_vid_preds.append(rgb_video_pred)
        rgb_frame_preds.append(rgb_frame_pred)
        flow_vid_preds.append(flow_video_pred)
        flow_frame_preds.append(flow_frame_pred)
        vid_lens.append(vid_len)
        labels.append(label)

    rgb_vid_preds = np.array(rgb_vid_preds)
    rgb_frame_preds = np.array(rgb_frame_preds)
    flow_vid_preds = np.array(flow_vid_preds)
    flow_frame_preds = np.array(flow_frame_preds)
    vid_lens = np.array(vid_lens)
    labels = np.array(labels)

    dmap, iou = dtmAP(rgb_vid_preds, flow_vid_preds, rgb_frame_preds, flow_frame_preds,
                      vid_lens, dataloader.dataset.path_to_annotations, args)

    sum = 0
    count = 0
    for item in list(zip(iou, dmap)):
        print('Detection map @ %f = %f' % (item[0], item[1]))
        if count < 7:
            sum = sum + item[1]
            count += 1

    print('average map = %f' % (sum / count))
    utils.write_results_to_eval_file(args, dmap, args.rgb_load_epoch, args.flow_load_epoch)
