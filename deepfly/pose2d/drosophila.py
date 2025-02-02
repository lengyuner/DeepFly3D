



'''
Namespace(acc_joints=[2, 3, 4, 7, 8, 9, 12, 13, 14], annotation_path='/home/guenel/public_html/drosophilaannotate/data/',
arch='hg', augmentation=True, blocks=1, carry=False, checkpoint='checkpoint',
csv_file_train=None, csv_file_val=None, data_folder='data/drosophila/', epochs=200,
features=128, gamma=0.1, hm_res=[64, 128], img_res=[256, 512], inplanes=64,
json_file='C:\\Users\\ps\\Desktop\\djz\\DeepFly3D_hubot\\deepfly\\pose2d\\../../data/drosophilaimaging-export.json',
lr=0.00025, momentum=0, multiview=False, name='', num_classes=19, num_output_image=0,
output_folder='df3d/', resume='./weights/sh8_mpii.tar', schedule=[25, 40, 70], sigma=1,
stacks=8, start_epoch=0, stride=2, test_batch=64, train_batch=6, train_folder_list=None,
train_joints=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
unlabeled='./data/test', unlabeled_recursive=False, weight_decay=0, workers=8)


# python deepfly/pose2d/drosophila.py --resume ./weights/sh8_deepfly.tar --unlabeled /home/user/Desktop/DeepFly3D/data/test
# python deepfly/pose2d/drosophila.py --resume ./weights/sh8_mpii.tar --unlabeled ./data/test
# python deepfly/pose2d/drosophila.py --resume ./weights/sh8_mpii.tar --unlabeled ./data/test
# python deepfly/pose2d/drosophila.py --resume ./weights/sh8_mpii.tar --unlabeled ./data/test

# python deepfly/pose2d/drosophila.py --resume ./weights/sh8_deepfly.tar --unlabeled /home/user/Desktop/DeepFly3D/data/test
# python deepfly/pose2d/drosophila.py --resume ./weights/sh8_mpii.tar --unlabeled ./data/test

# python test.py --resume ./weights/sh8_mpii.tar --unlabeled ./data/test

conda activate deepfly3
cd Desktop
cd djz

cd DeepFly3D_hubot

python test.py --resume ./weights/sh8_mpii.tar --unlabeled ./data/test
python pose2d/drosophila.py --resume ./weights/sh8_deepfly.tar --unlabeled /home/user/Desktop/DeepFly3D/data/test

'''



from __future__ import print_function, absolute_import

import sys
sys.path.append('..')
import time

import matplotlib as mpl

# mpl.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from deepfly.pose2d.progress.progress.bar import Bar, NoOutputBar
from deepfly.pose2d.utils.logger import Logger, savefig
from deepfly.pose2d.utils.evaluation import accuracy, AverageMeter, mse_acc
from deepfly.pose2d.utils.misc import save_checkpoint, save_dict
from deepfly.pose2d.utils.osutils import isfile, join, find_leaf_recursive
from deepfly.pose2d.utils.imutils import save_image, drosophila_image_overlay
from deepfly.pose2d.ArgParse import create_parser
from deepfly.GUI.util.os_util import *
import deepfly.pose2d.datasets
import deepfly.pose2d.models as models
from deepfly.pose2d.utils.osutils import mkdir_p, isdir
import os
from deepfly.pose2d.utils.misc import get_time, to_numpy
from deepfly.GUI.Camera import Camera
from deepfly.GUI.Config import config
from pathlib import Path

from logging import getLogger
import logging
import cv2

import pdb

# print('Import successfully')
# print('Current workspace is: ', os.getcwd())

best_acc = 0

def weighted_mse_loss(inp, target, weights):
    out = (inp - target) ** 2
    out = out * weights.expand_as(out)
    loss = out.sum()

    return loss


def main(args):

    global best_acc

    # create model
    getLogger('df3d').debug("Creating model '{}', stacks={}, blocks={}".format(
            args.arch, args.stacks, args.blocks
        )
    )
    model = models.__dict__[args.arch](
        num_stacks=args.stacks,
        num_blocks=args.blocks,
        num_classes=args.num_classes,
        num_feats=args.features,
        inplanes=args.inplanes,
        init_stride=args.stride,
    )

    model = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.MSELoss(reduction='mean').cuda()  # deprecated: size_average=True
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=5
    )

    # optionally resume from a checkpoint
    title = "Drosophila-" + args.arch
    # print(isfile(args.resume),'args.resume','^'*100)
    if args.resume:
        if isfile(args.resume):
            getLogger('df3d').debug("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if "mpii" in args.resume and not args.unlabeled:  # weights for sh trained on mpii dataset
                getLogger('df3d').debug("Removing input/output layers")
                ignore_weight_list_template = [
                    "module.score.{}.bias",
                    "module.score.{}.weight",
                    "module.score_.{}.weight",
                ]
                ignore_weight_list = list()
                for i in range(8):
                    for template in ignore_weight_list_template:
                        ignore_weight_list.append(template.format(i))
                for k in ignore_weight_list:
                    if k in checkpoint["state_dict"]:
                        checkpoint["state_dict"].pop(k)

                state = model.state_dict()
                state.update(checkpoint["state_dict"])
                getLogger('df3d').debug(model.state_dict())
                getLogger('df3d').debug(checkpoint["state_dict"])
                model.load_state_dict(state, strict=False)
            elif "mpii" in args.resume and args.unlabeled:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                # print("$"*100)
            else:
                pretrained_dict = checkpoint["state_dict"]
                model.load_state_dict(pretrained_dict, strict=True)

                args.start_epoch = checkpoint["epoch"]
                args.img_res = checkpoint["image_shape"]
                args.hm_res = checkpoint["heatmap_shape"]
                getLogger('df3d').debug("Loading the optimizer")
                getLogger('df3d').debug(
                    "Setting image resolution and heatmap resolution: {} {}".format(
                        args.img_res, args.hm_res
                    )
                )

            getLogger('df3d').debug(
                "Loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise FileNotFoundError

    # prepare loggers
    if not args.unlabeled:
        logger = Logger(join(args.checkpoint, "log.txt"), title=title)
        logger.set_names(
            [
                "Epoch",
                "LR",
                "Train Loss",
                "Val Loss",
                "Train Acc",
                "Val Acc",
                "Val Mse",
                "Val Jump",
            ]
        )

    # cudnn.benchmark = True
    getLogger('df3d').debug("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # print(args.unlabeled,'unlabeled','$'*100)
    # if args.unlabeled:
    #     print(args.unlabeled, 'unlabeled', '$' * 100)
    if args.unlabeled:
        if args.unlabeled[0] == '/': # wtf why does it have a slash before it where did that come from?
            args.unlabeled = args.unlabeled[1:]
        unlabeled_folder = args.unlabeled
        print("UNLABELED FOLDER:")
        print(unlabeled_folder)

        max_img_id = get_max_img_id(unlabeled_folder)


        try:
            max_img_id = min(max_img_id, args.num_images_max-1)
            # print(max_img_id, 'try','$' * 100)
        except:
            pass
        getLogger('df3d').debug('Going to process {} images'.format(max_img_id+1))
        # print(unlabeled_folder, '$' * 100)
        unlabeled_loader = DataLoader(
            deepfly.pose2d.datasets.Drosophila(
                data_folder=args.data_folder,
                train=False,
                sigma=args.sigma,
                session_id_train_list=None,
                folder_train_list=None,
                img_res=args.img_res,
                hm_res=args.hm_res,
                augmentation=False,
                evaluation=True,
                unlabeled=unlabeled_folder,
                num_classes=args.num_classes,
                max_img_id=max_img_id,
            ),
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=False,
        )
        # print(unlabeled_folder, 'before validate','$' * 100)
        valid_loss, valid_acc, val_pred, val_score_maps, mse, jump_acc = validate(
            unlabeled_loader, 0, model, criterion, args, save_path=unlabeled_folder
        )
        # print(unlabeled_folder, 'after validate', '$' * 100)
        # print(unlabeled_folder, '$' * 100)
        unlabeled_folder_replace = 'data_test'#os.path.join(unlabeled_folder, 'a')

        # print(unlabeled_folder_replace, '$' * 100)
        # unlabeled_folder_replace = unlabeled_folder.replace("/", "-") # TODO(JZ)
        # print(unlabeled_folder_replace, 'unlabeled_folder_replace','$' * 100)
        getLogger('df3d').debug(f"val_score_maps have shape: {val_score_maps.shape}")

        getLogger('df3d').debug("Saving Results, flipping heatmaps")
        cid_to_reverse = config["flip_cameras"]  # camera id to reverse predictions and heatmaps
        # cidread2cid, cid2cidread = read_camera_order(os.path.join(unlabeled_folder, 'df3d')) # TODO
        cidread2cid, cid2cidread = read_camera_order(unlabeled_folder)  # TODO
        # cidread2cid, cid2cidread = read_camera_order(os.path.join('./data/temp', 'df3d'))
        # print(cidread2cid, '\n',cid2cidread, '$' * 100)
        cid_read_to_reverse = [cid2cidread[cid] for cid in cid_to_reverse]
        getLogger('df3d').debug(
            "Flipping heatmaps for images with cam_id: {}".format(
                cid_read_to_reverse
            )
        )
        val_pred[cid_read_to_reverse, :, :, 0] = (
            1 - val_pred[cid_read_to_reverse, :, :, 0]
        )
        # print('$' * 100)
        for cam_id in cid_read_to_reverse:
            for img_id in range(val_score_maps.shape[1]):
                for j_id in range(val_score_maps.shape[2]):
                    val_score_maps[cam_id, img_id, j_id, :, :] = cv2.flip(
                        val_score_maps[cam_id, img_id, j_id, :, :], 1
                    )
        # print(val_pred.shape)
        print(os.path.join(args.unlabeled,"preds_{}.pkl".format(unlabeled_folder_replace) ))

        save_dict(
            val_pred,
            os.path.join(
                args.unlabeled,
                "preds_{}.pkl".format(unlabeled_folder_replace),
                # args.data_folder,
                # "{}".format(unlabeled_folder),
                # args.output_folder,
                # "./preds_{}.pkl".format(unlabeled_folder_replace), # TODO(JZ)
            ),
        )


        # print(val_score_maps.shape)
        print(os.path.join(args.unlabeled,"score_maps_{}.pkl".format(unlabeled_folder_replace) ))


        save_dict(
            val_score_maps,
            os.path.join(
                args.unlabeled,
                "score_maps_{}.pkl".format(unlabeled_folder_replace),
                # args.data_folder,
                # "{}".format(unlabeled_folder),
                # args.output_folder,
                # "./preds_{}.pkl".format(unlabeled_folder_replace), # TODO(JZ)
            ),
        )

        getLogger('df3d').debug("Finished saving results")
        # print(unlabeled_folder_replace)

    else:
        train_loader, val_loader = create_dataloader()
        lr = args.lr
        for epoch in range(args.start_epoch, args.epochs):
            # lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
            getLogger('df3d').debug("\nEpoch: %d | LR: %.8f" % (epoch + 1, lr))

            # train for one epoch
            train_loss, train_acc, train_predictions, train_mse, train_mse_jump = train(
                train_loader, epoch, model, optimizer, criterion, args
            )
            # # evaluate on validation set
            valid_loss, valid_acc, val_pred, val_score_maps, mse, jump_acc = validate(
                val_loader, epoch, model, criterion, args, save_path=args.unlabeled
            )
            scheduler.step(valid_loss)
            # append logger file
            logger.append(
                [
                    epoch + 1,
                    lr,
                    train_loss,
                    valid_loss,
                    train_acc,
                    valid_acc,
                    mse,
                    jump_acc,
                ]
            )

            # remember best acc and save checkpoint
            is_best = valid_acc > best_acc
            best_acc = max(valid_acc, best_acc)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "multiview": args.multiview,
                    "image_shape": args.img_res,
                    "heatmap_shape": args.hm_res,
                },
                val_pred,
                is_best,
                checkpoint=args.checkpoint,
            )

            fig = plt.figure()
            logger.plot(["Train Acc", "Val Acc"])
            savefig(os.path.join(args.checkpoint, "log.eps"))
            plt.close(fig)
    # print('finished','$'*100)
    return val_score_maps, val_pred
    # logger.close()


def train(train_loader, epoch, model, optimizer, criterion, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    mse = AverageMeter()
    mse_hip = AverageMeter()
    mse_coxa = AverageMeter()
    mse_femur = AverageMeter()
    mse_tibia = AverageMeter()
    mse_tarsus = AverageMeter()
    mse_jump = AverageMeter()

    avg_local_max = AverageMeter()

    # to later save the disk
    predictions = dict()
    # switch to train mode
    model.train()

    end = time.time()

    bar = Bar("Processing", max=len(train_loader))# if logging.getLogger('df3d').isEnabledFor(logging.DEBUG) else NoOutputBar()
    bar.start()
    for i, (inputs, target, meta) in enumerate(train_loader):
        # reset seed for truly random transformations on input
        np.random.seed()
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = (
            torch.autograd.Variable(inputs.cuda())
            if torch.cuda.is_available()
            else torch.autograd.Variable(inputs)
        )
        target_var = torch.autograd.Variable(target.cuda(non_blocking=True))

        # compute output
        output = model(input_var)
        score_map = output[-1].data.cpu()

        loss_weight = torch.ones((inputs.size(0), args.num_classes, 1, 1))
        if np.any(np.array(meta["joint_exists"]) == 0):
            batch_index = (np.logical_not(meta["joint_exists"])).nonzero()[:, 0]
            joint_index = (np.logical_not(meta["joint_exists"])).nonzero()[:, 1]
            loss_weight[batch_index, joint_index, :, :] = 0.0

        # logger.debug(loss_weight)
        loss_weight.requires_grad = True
        loss_weight = loss_weight.cuda()
        loss = weighted_mse_loss(output[0], target_var, weights=loss_weight)
        for j in range(1, len(output)):
            loss += weighted_mse_loss(output[j], target_var, weights=loss_weight)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(score_map, target, args.acc_joints)

        # measure accuracy and record loss
        mse_err = mse_acc(target_var.data.cpu(), score_map)

        getLogger('df3d').debug(f'{mse_err.size()}, {mse_err[:, 0] > 50}, {meta["image_name"][0]}')
        mse.update(torch.mean(mse_err[args.acc_joints, :]), inputs.size(0))  # per joint
        mse_hip.update(torch.mean(mse_err[np.arange(0, 15, 5), :]), inputs.size(0))
        mse_coxa.update(torch.mean(mse_err[np.arange(1, 15, 5), :]), inputs.size(0))
        mse_femur.update(torch.mean(mse_err[np.arange(2, 15, 5), :]), inputs.size(0))
        mse_tibia.update(torch.mean(mse_err[np.arange(3, 15, 5), :]), inputs.size(0))
        mse_tarsus.update(torch.mean(mse_err[np.arange(4, 15, 5), :]), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        jump_thr = args.hm_res[0] * 5.0 / 64.0
        mse_jump.update(
            np.mean(np.ravel(mse_err[args.acc_joints, :] > jump_thr)) * 100.0,
            inputs.size(0),
        )

        if i < args.num_output_image:
            drosophila_img = drosophila_image_overlay(
                inputs, target, args.hm_res, 3, args.train_joints
            )

            folder_name = (
                meta["folder_name"][0].replace("/", "-")
                + meta["folder_name"][1].replace("/", "-")
                + meta["folder_name"][2].replace("/", "-")
            )
            image_name = meta["image_name"][0]
            template = "./train_gt_{}_{}_{:06d}_{}_joint_{}.jpg"

            save_image(
                img_path=os.path.join(
                    args.checkpoint_image_dir,
                    template.format(
                        epoch, folder_name, meta["pid"][0], meta["cid"][0], []
                    ),
                ),
                img=drosophila_img,
            )

        if i < args.num_output_image and False:
            drosophila_img = drosophila_image_overlay(
                inputs,
                score_map,
                args.hm_res,
                3,
                args.train_joints,
                img_id=int(meta["image_name"][0].split("_")[-1]),
            )
            template = "{}_train_pred_epoch_{}_{}_{:06d}_{}.jpg"

            folder_name = meta["folder_name"][0]
            folder_name = (
                folder_name.replace(" ", "_").replace("\\", "-").replace("/", "-")
            )
            save_image_name = template.format(
                "left" if meta["cid"][0] == 0 else "right",
                epoch,
                folder_name,
                meta["pid"][0],
                meta["cid"][0],
            )

            save_image_name = save_image_name.replace(" ", "_ ").replace("\\", "-")
            save_image(
                img_path=os.path.join(args.checkpoint_image_dir, save_image_name),
                img=drosophila_img,
            )

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) D:{data:.6f}s|B:{bt:.3f}s|L:{loss:.8f}|Acc:{acc: .4f}|Mse:{mse: .3f} F-T:{mse_femur: .3f} T-Tar:{mse_tibia: .3f} Tar-tip:{mse_tarsus: .3f} Jump:{mse_jump: .4f}".format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg,
            mse=mse.avg,
            mse_hip=mse_hip.avg,
            mse_femur=mse_femur.avg,
            mse_tibia=mse_tibia.avg,
            mse_tarsus=mse_tarsus.avg,
            mse_coxa=mse_coxa.avg,
            mse_jump=mse_jump.avg,
        )
        bar.next()

    bar.finish()
    return losses.avg, acces.avg, predictions, mse.avg, mse_jump.avg


def validate(val_loader, epoch, model, criterion, args, save_path=False):
    # keeping statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    mse = AverageMeter()
    mse_hip = AverageMeter()
    mse_coxa = AverageMeter()
    mse_femur = AverageMeter()
    mse_tibia = AverageMeter()
    mse_tarsus = AverageMeter()
    mse_jump = AverageMeter()
    avg_local_max = AverageMeter()

    # predictions and score maps
    num_cameras = 7
    predictions = np.zeros(
        shape=(
            num_cameras + 1,
            val_loader.dataset.greatest_image_id() + 1,
            config["num_predict"],
            2,
        ),
        dtype=np.float32,
    )  # num_cameras+1 for the mirrored camera 3
    getLogger('df3d').debug("Predictions shape {}".format(predictions.shape))

    if save_path is not None:
        print("SAVE_PATH:")
        print(save_path)
        unlabeled_folder_replace = 'data_test' #save_path.replace("/", "-")  # TODO(JZ)
        # score_map_filename = os.path.join(args.data_folder, "./heatmap_{}.pkl".format(unlabeled_folder_replace))
        score_map_filename = os.path.join(
            # args.data_folder,
            "{}".format(save_path),
            # args.output_folder,
            "heatmap_{}.pkl".format(unlabeled_folder_replace),
        )

        # save_dict(
        #     val_score_maps,
        #     os.path.join(
        #         args.unlabeled,
        #         "output/score_maps_{}.pkl".format(unlabeled_folder_replace),
        #         # args.data_folder,
        #         # "{}".format(unlabeled_folder),
        #         # args.output_folder,
        #         # "./preds_{}.pkl".format(unlabeled_folder_replace), # TODO(JZ)
        #     ),
        # )
        print(score_map_filename,'\nscore_map_filename\n','$'*70)

        score_map_path = Path(score_map_filename)
        score_map_path.parent.mkdir(exist_ok=True, parents=True)
        score_map_arr = np.memmap(
            score_map_filename,
            dtype="float32",
            mode="w+",
            shape=(
                num_cameras + 1,
                val_loader.dataset.greatest_image_id() + 1,
                config["num_predict"],
                args.hm_res[0],
                args.hm_res[1],
            ),
        )  # num_cameras+1 for the mirrored camera 3

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar("Processing", max=len(val_loader)) #if logging.getLogger('df3d').isEnabledFor(logging.INFO) else NoOutputBar()
    bar.start()
    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
        score_map = output[-1].data.cpu()

        # loss = criterion(output[-1], target_var)
        acc = accuracy(score_map.cpu(), target.cpu(), args.acc_joints)

        # generate predictions
        if save_path is not None:
            for n, cam_id, cam_read_id, img_id in zip(
                range(score_map.size(0)), meta["cid"], meta["cam_read_id"], meta["pid"]
            ):
                smap = to_numpy(score_map[n, :, :, :])
                score_map_arr[cam_read_id, img_id, :] = smap
                pr = Camera.hm_to_pred(
                    score_map_arr[cam_read_id, img_id, :], threshold_abs=0.0
                )
                predictions[cam_read_id, img_id, :] = pr

        # measure accuracy and record loss
        mse_err = mse_acc(target_var.data.cpu(), score_map)
        mse.update(torch.mean(mse_err[args.acc_joints, :]), inputs.size(0))
        mse_hip.update(
            torch.mean(mse_err[np.arange(0, mse_err.shape[0], 5), :]), inputs.size(0)
        )
        mse_coxa.update(
            torch.mean(mse_err[np.arange(1, mse_err.shape[0], 5), :]), inputs.size(0)
        )
        mse_femur.update(
            torch.mean(mse_err[np.arange(2, mse_err.shape[0], 5), :]), inputs.size(0)
        )
        mse_tibia.update(
            torch.mean(mse_err[np.arange(3, mse_err.shape[0], 5), :]), inputs.size(0)
        )
        mse_tarsus.update(
            torch.mean(mse_err[np.arange(4, mse_err.shape[0], 5), :]), inputs.size(0)
        )

        losses.update(0, inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        jump_thr = args.hm_res[0] * 5.0 / 64.0
        mse_jump.update(
            np.mean(np.ravel(mse_err[args.acc_joints, :] > jump_thr)) * 100.0,
            inputs.size(0),
        )

        if args.unlabeled and i < args.num_output_image:
            drosophila_img = drosophila_image_overlay(
                inputs,
                score_map,
                args.hm_res,
                3,
                np.arange(config["num_predict"]),
                img_id=int(meta["image_name"][0].split("_")[-1]),
            )

            template = "{}_overlay_val_epoch_{}_{}_{:06d}_{}.jpg"

            folder_name = meta["folder_name"][0]
            folder_name = (
                folder_name.replace(" ", "_").replace("\\", "-").replace("/", "-")
            )
            save_image_name = template.format(
                "left" if meta["cid"][0] == 0 else "right",
                epoch,
                folder_name,
                meta["pid"][0],
                meta["cid"][0],
            )

            save_image_name = save_image_name.replace(" ", "_ ").replace("\\", "-")
            save_image(
                img_path=os.path.join(args.checkpoint_image_dir, save_image_name),
                img=drosophila_img,
            )

        elif i < args.num_output_image:
            drosophila_img = drosophila_image_overlay(
                inputs, score_map, args.hm_res, 3, np.arange(config["num_predict"])
            )
            folder_name = meta["folder_name"][0]

            template = "./overlay_val_epoch_{}_{}_{:06d}_{}.jpg"
            save_image_name = template.format(
                epoch, folder_name.replace("/", "-"), meta["pid"][0], meta["cid"][0]
            )

            save_image_name = save_image_name.replace(" ", "_ ").replace("\\", "-")
            save_image(
                img_path=os.path.join(args.checkpoint_image_dir, save_image_name),
                img=drosophila_img,
            )

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) D:{data:.6f}s|B:{bt:.3f}s|L:{loss:.8f}|Acc:{acc: .4f}|Mse:{mse: .3f}|F-T:{mse_femur: .3f}|T-Tar:{mse_tibia: .3f}|Tar-tip:{mse_tarsus: .3f}".format(
            batch=i + 1,
            size=len(val_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg,
            mse=mse.avg,
            mse_hip=mse_hip.avg,
            mse_femur=mse_femur.avg,
            mse_tibia=mse_tibia.avg,
            mse_tarsus=mse_tarsus.avg,
            mse_coxa=mse_coxa.avg,
            mse_jump=mse_jump.avg,
            local_max_count=avg_local_max.avg,
        )
        bar.next()

    bar.finish()
    if save_path is not None:
        score_map_arr.flush()
    else:
        score_map_arr = None
    return losses.avg, acces.avg, predictions, score_map_arr, mse.avg, mse_jump.avg


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def create_dataloader():
    session_id_list = [
        "q47rx0Ybo0QHraRuDWken9WtPTA2"
    ]
    train_session_id_list, test_session_id_list = session_id_list, session_id_list
    if args.train_folder_list is None:
        args.train_folder_list = [
            "data/labeled/right_side_short",
        ]
    test_folder_list = ["data/labeled/right_side_short"]
    # make sure training and test sets are mutually exclusive
    #assert (
    #    len(set.intersection(set(args.train_folder_list), set(test_folder_list))) == 0
    #)

    train_loader = DataLoader(
        deepfly.pose2d.datasets.Drosophila(
            data_folder=args.data_folder,
            train=True,
            sigma=args.sigma,
            session_id_train_list=train_session_id_list,
            folder_train_list=args.train_folder_list,
            img_res=args.img_res,
            hm_res=args.hm_res,
            augmentation=args.augmentation,
            num_classes=args.num_classes,
            jsonfile=args.json_file,
            csvfile=args.csv_file_train,
        ),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        deepfly.pose2d.datasets.Drosophila(
            data_folder=args.data_folder,
            train=False,
            sigma=args.sigma,
            session_id_train_list=test_session_id_list,
            folder_train_list=test_folder_list,
            img_res=args.img_res,
            hm_res=args.hm_res,
            augmentation=False,
            evaluation=True,
            num_classes=args.num_classes,
            jsonfile=args.json_file,
            csvfile=args.csv_file_val,
        ),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    return train_loader, val_loader


# Code belows trains the network starting from the MPII dataset
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.train_joints = np.arange(args.num_classes)
    acc_joints_tmp = []
    for i in range(3):
        p = [(i * 5) + k for k in args.acc_joints]
        acc_joints_tmp.extend(p)
    args.acc_joints = acc_joints_tmp

    # print(args.acc_joints, '^'*100) #TODO
    args.num_images_max = 1
    args.workers = 2

    getLogger('df3d').setLevel(10)
    getLogger('df3d').debug(f"Training joints: {args.train_joints}")
    getLogger('df3d').debug(f"Acc joints: {args.acc_joints}")

    args.checkpoint = os.path.join(
        args.checkpoint,
        get_time()
        + "_{}_{}_{}_{}_{}_{}_{}".format(
            "predict" if args.unlabeled else "training",
            args.arch,
            args.stacks,
            args.img_res,
            args.blocks,
            "mv" if args.multiview else "",
            "temp" if args.multiview else "",
            args.name,
        ),
    )
    args.checkpoint = (
        args.checkpoint.replace(" ", "_").replace("(", "_").replace(")", "_")
    )
    args.checkpoint = args.checkpoint.replace("__", "_").replace("--", "-")
    getLogger('df3d').debug("Checkpoint dir: {}".format(args.checkpoint))
    args.checkpoint_image_dir = os.path.join(args.checkpoint, "./images/")

    # create checkpoint dir and image dir
    if args.unlabeled is None:
        if not isdir(args.checkpoint):
            mkdir_p(args.checkpoint)
        if not isdir(args.checkpoint_image_dir):
            mkdir_p(args.checkpoint_image_dir)
        if args.carry and not isdir(
            os.path.join(args.annotation_path, args.unlabeled + "_network")
        ):
            mkdir_p(os.path.join(args.annotation_path, args.unlabeled + "_network"))
        if args.carry and not isdir(
            os.path.join(args.data_folder, args.unlabeled + "_network")
        ):
            mkdir_p(os.path.join(args.data_folder, args.unlabeled + "_network"))

    print(args, '\n' * 3)
    main(args)
    # logger.close()

# woziji kankna ba
# ganjue yishi banhui gaobu chulai
# zhege ren de daima shizai shi tai shile







