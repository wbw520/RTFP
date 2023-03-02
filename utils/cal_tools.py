import numpy as np


class IouCal(object):
    def __init__(self, args):
        self.num_class = args.num_classes
        self.hist = np.zeros((self.num_class, self.num_class))
        if args.dataset == "cityscapes":
            self.name = ["road:", "sidewalk:", "building:", "wall:", "fence:", "pole:", "traffic light:", "traffic sign:",
                         "vegetation:", "terrain:", "sky:", "person:", "rider:", "car:", "truck:", "bus:", "train:",
                         "motorcycle:", "bicycle:"]
        elif args.dataset == "facade":
            self.name = ["BG:", "building:", "window:", "sky:", "roof:", "door:", "tree:", "people:", "car:", "sign:"]

    def fast_hist(self, label, pred, num_class):
        k = (label >= 0) & (pred < self.num_class)
        return np.bincount(num_class * label[k].astype(int) + pred[k], minlength=num_class ** 2).reshape(num_class, num_class)

    def per_class_iou(self, hist):
        return np.diag(hist)/(hist.sum(1) + hist.sum(0) - np.diag(hist))   # IOU = TP / (TP + FP + FN)

    def evaluate(self, labels, preds):
        labels = labels.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        for label, pred in zip(labels, preds):
            self.hist += self.fast_hist(label.flatten(), pred.flatten(), self.num_class)

    def iou_demo(self):
        hist2 = np.zeros((self.num_class - 1, self.num_class - 1))
        for s in range(self.num_class - 1):
            for k in range(self.num_class - 1):
                hist2[s][k] = self.hist[s + 1][k + 1]

        self.hist = hist2
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

        iou = self.per_class_iou(self.hist)
        STR = ""
        for i in range(len(self.name) - 1):
            STR = STR + self.name[i+1] + str(round(iou[i], 3)) + " "
        print(STR)
        miou = np.nanmean(iou)

        return round(acc, 3), round(acc_cls, 3), round(miou, 3)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'