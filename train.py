import argparse
import sys
sys.path.append("..")

import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

import os
import os.path as osp
from DualNet import DualNet
from BraTSDataSet import BraTSDataSet, BraTSValDataSet, my_collate
import timeit
from tensorboardX import SummaryWriter
import loss_Dual as loss
from engine import Engine
from math import ceil

from DualNet import conv3x3x3

start = timeit.default_timer()
kd_wt = 0.1


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Shared-Specific model for 3D Medical Image Segmentation.")

    parser.add_argument("--data_dir", type=str, default='./datalist/')
    parser.add_argument("--train_list", type=str, default='BraTS20/BraTS20_train.csv')
    parser.add_argument("--val_list", type=str, default='BraTS20_val.csv')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/example/')
    parser.add_argument("--reload_path", type=str, default='snapshots/example/last.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=False)
    parser.add_argument("--random_scale", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=999)

    parser.add_argument("--norm_cfg", type=str, default='IN')  # normalization
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--mode", type=str, default='0,1,2,3')
    parser.add_argument("--teachers", type=str, default='0')

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2*num / den

    return dice.mean()


def compute_dice_score(preds, labels):

    preds = F.sigmoid(preds)

    pred_ET = preds[:, 0, :, :, :]
    pred_WT = preds[:, 1, :, :, :]
    pred_TC = preds[:, 2, :, :, :]
    label_ET = labels[:, 0, :, :, :]
    label_WT = labels[:, 1, :, :, :]
    label_TC = labels[:, 2, :, :, :]
    dice_ET = dice_score(pred_ET, label_ET).cpu().data.numpy()
    dice_WT = dice_score(pred_WT, label_WT).cpu().data.numpy()
    dice_TC = dice_score(pred_TC, label_TC).cpu().data.numpy()
    return dice_ET, dice_WT, dice_TC


def predict_sliding(args, net, imagelist, tile_size, classes, mode):
    image, image_res = imagelist
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    full_probs = torch.from_numpy(full_probs).cuda()
    count_predictions = torch.from_numpy(count_predictions).cuda()

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                # img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]

                prediction, _, _ = net(img, val=True, mode=mode)

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs


def validate(args, input_size, model, ValLoader, num_classes):
    # start to validate
    val_ET = [0, 0, 0, 0]
    val_WT = [0, 0, 0, 0]
    val_TC = [0, 0, 0, 0]

    # return 2, 0, 2

    for index, batch in enumerate(ValLoader):
        # print('validation %d processd'%(index))
        image, image_res, label, size, name, affine = batch
        image = image.cuda()
        image_res = image_res.cuda()
        label = label.cuda()
        with torch.no_grad():
            for i_mode in [0, 1, 2, 3]:
                pred = predict_sliding(args, model, [image, image_res], input_size, num_classes, mode=str(i_mode))
                dice_ET, dice_WT, dice_TC = compute_dice_score(pred, label)
                val_ET[i_mode] += dice_ET
                val_WT[i_mode] += dice_WT
                val_TC[i_mode] += dice_TC

    print('Val_ET_Dice:', val_ET)
    print('Val_WT_Dice:', val_WT)
    print('Val_TC_Dice:', val_TC)
    return val_ET.index(max(val_ET)), val_WT.index(max(val_WT)), val_TC.index(max(val_TC))


def main():
    """Create the ConResNet model and then start the training."""
    parser = get_arguments()
    print(parser)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                        num_classes=args.num_classes, weight_std=args.weight_std, self_att=False, cross_att=False)
        model.train()
        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                # model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
                checkpoint = torch.load(args.reload_path)
                model = checkpoint['model']
                optimizer = checkpoint['optimizer']
                args.start_iters = checkpoint['iter']
                print("Loaded model trained for", args.start_iters, "iters")
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))
                exit(0)

        loss_D = loss.DiceLoss4BraTS().to(device)
        loss_BCE = loss.BCELoss4BraTS().to(device)

        print('current mode:', args.mode)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        trainloader, train_sampler = engine.get_train_loader(BraTSDataSet(args.data_dir, args.train_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror), collate_fn=my_collate)
        valloader, val_sampler = engine.get_test_loader(BraTSValDataSet(args.data_dir, args.val_list))

        val_Dice_best = -999999
        teachers = args.teachers
        for i_iter, batch in enumerate(trainloader):
            i_iter += args.start_iters
            images = torch.from_numpy(batch['image']).cuda()
            labels = torch.from_numpy(batch['label']).cuda()

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)

            preds, mode_split, totKDLoss = model(images, mode=args.mode, teachers=teachers)
            preds_seg = preds

            term_seg_Dice = loss_D.forward(preds_seg, labels)
            term_seg_BCE = loss_BCE.forward(preds_seg, labels)

            term_all = term_seg_Dice + term_seg_BCE + kd_wt * totKDLoss
            term_all.backward()

            optimizer.step()

            if i_iter % 100 == 0 and (args.local_rank == 0):
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss', term_all.cpu().data.numpy(), i_iter)

            print('iter = {} of {} completed, lr = {:.4}, seg_loss = {:.4}, kd_loss = {:.4}'
                .format(i_iter, args.num_steps, lr, (term_seg_Dice+term_seg_BCE).cpu().data.numpy(), totKDLoss.cpu().data.numpy()))

            if i_iter >= args.num_steps - 1 and (args.local_rank == 0):
                print('save last model ...')
                checkpoint = {
                    'model': model,
                    'optimizer': optimizer,
                    'iter': i_iter
                }
                torch.save(checkpoint, osp.join(args.snapshot_dir, 'final.pth'))
                break

            if i_iter % args.val_pred_every == args.val_pred_every - 1 and i_iter != 0 and (args.local_rank == 0):
                print('save model ...')
                checkpoint = {
                    'model': model,
                    'optimizer': optimizer,
                    'iter': i_iter
                }
                # torch.save(checkpoint, osp.join(args.snapshot_dir, 'iter_' + str(i_iter) + '.pth'))
                torch.save(checkpoint, osp.join(args.snapshot_dir, 'last.pth'))

            # val and identify the best modality for each tumor
            if not args.train_only and i_iter % args.val_pred_every == 0:
                print('validate ...')
                teacher_modes = validate(args, input_size, model, valloader, args.num_classes)
                # teachers = [max(teacher_modes, key=teacher_modes.count)]  # single teacher
                teachers = list(set(teacher_modes))  # multi-teacher
                teachers = ",".join(map(str, teachers))
                print('teachers:', teachers)

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
