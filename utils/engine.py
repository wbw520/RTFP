from math import ceil
import torch
import numpy as np
import math
import sys
import utils2.lr_sched as lr_sched
from utils.cal_tools import IouCal, AverageMeter, ProgressMeter


def train_model(args, epoch, model, train_loader, criterion, optimizer, loss_scaler, device):
    model.train()
    train_main_loss = AverageMeter('Train Main Loss', ':.5')
    train_aux_loss = AverageMeter('Train Aux Loss', ':.5')
    lr = AverageMeter('lr', ':.5')
    L = len(train_loader)
    curr_iter = epoch * L
    progress = ProgressMeter(L, [train_main_loss, train_aux_loss, lr], prefix="Epoch: [{}]".format(epoch))
    accum_iter = args.accum_iter

    for data_iter_step, data in enumerate(train_loader):
        optimizer.param_groups[0]['lr'] = args.lr * (1 - float(curr_iter) / (args.num_epoch * L)) ** args.lr_decay
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)

        inputs = data["images"].to(device, dtype=torch.float32)
        mask = data["masks"].to(device, dtype=torch.int64)

        with torch.cuda.amp.autocast():
            outputs, aux = model(inputs)
            main_loss = criterion(outputs, mask)
            aux_loss = criterion(aux, mask)
            loss = main_loss + 0.4 * aux_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        train_main_loss.update(main_loss.item())
        train_aux_loss.update(aux_loss.item())
        lr.update(optimizer.param_groups[0]['lr'])

        curr_iter += 1

        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)


def evaluation(args, best_record, epoch, model, model_without_ddp, val_loader, criterion, device):
    model.eval()
    val_loss = AverageMeter('Val Main Loss', ':.4')
    progress = ProgressMeter(len(val_loader), [val_loss], prefix="Epoch: [{}]".format(epoch))
    iou = IouCal(args)
    for i_batch, data in enumerate(val_loader):
        inputs = data["images"].to(device, dtype=torch.float32)
        mask = data["masks"].to(device, dtype=torch.int64)

        pred, full_pred = inference_sliding(args, model, inputs)
        iou.evaluate(pred, mask)
        val_loss.update(criterion(full_pred, mask).item())

        if i_batch % args.print_freq == 0:
            progress.display(i_batch)

    acc, acc_cls, mean_iou = iou.iou_demo()

    if mean_iou > best_record['mean_iou']:
        best_record['val_loss'] = val_loss.avg
        best_record['epoch'] = epoch
        best_record['acc'] = acc
        best_record['acc_cls'] = acc_cls
        best_record['mean_iou'] = mean_iou
        if args.output_dir:
            torch.save(model_without_ddp.state_dict(), args.output_dir + "_epoch" + str(epoch) + "_PSPNet.pt")

    print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iou %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iou))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iou %.5f], ---- [epoch %d], '
          % (best_record['val_loss'], best_record['acc'],
                    best_record['acc_cls'], best_record['mean_iou'], best_record['epoch']))

    print('-----------------------------------------------------------------------------------------------------------')


@torch.no_grad()
def inference_sliding(args, model, image):
    image_size = image.size()
    stride = int(math.ceil(args.crop_size[0] * args.stride_rate))
    tile_rows = ceil((image_size[2]-args.crop_size[0])/stride) + 1
    tile_cols = ceil((image_size[3]-args.crop_size[1])/stride) + 1
    b = image_size[0]

    full_probs = torch.from_numpy(np.zeros((b, args.num_classes, image_size[2], image_size[3]))).to(args.device)
    count_predictions = torch.from_numpy(np.zeros((b, args.num_classes, image_size[2], image_size[3]))).to(args.device)

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = x1 + args.crop_size[0]
            y2 = y1 + args.crop_size[1]
            if row == tile_rows - 1:
                y2 = image_size[2]
                y1 = image_size[2] - args.crop_size[1]
            if col == tile_cols - 1:
                x2 = image_size[3]
                x1 = image_size[3] - args.crop_size[0]

            img = image[:, :, y1:y2, x1:x2]

            with torch.set_grad_enabled(False):
                padded_prediction = model(img)
            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += padded_prediction  # accumulate the predictions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    _, preds = torch.max(full_probs, 1)
    return preds, full_probs