import numpy as np


class IouCal(object):
    def __init__(self, args):
        self.num_class = args.num_classes
        self.name = ["road:", "sidewalk:", "building:", "wall:", "fence:", "pole:", "traffic light:", "traffic sign:",
                     "vegetation:", "terrain:", "sky:", "person:", "rider:", "car:", "truck:", "bus:", "train:",
                     "motorcycle:", "bicycle:"]

    def fast_hist(self, label, pred, num_class):
        k = (label >= 0) & (pred < self.num_class)
        return np.bincount(num_class * label[k].astype(int) + pred[k], minlength=num_class ** 2).reshape(num_class, num_class)

    def per_class_iou(self, hist):
        return np.diag(hist)/(hist.sum(1) + hist.sum(0) - np.diag(hist))   # IOU = TP / (TP + FP + FN)

    def evaluate(self, labels, preds):
        hist = np.zeros((self.num_class, self.num_class))
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())
        for label, pred in zip(labels, preds):
            hist += self.fast_hist(label.flatten(), pred.flatten(), self.num_class)
        return hist

    def iou_demo(self, labels, preds):
        hist = self.evaluate(labels, preds)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

        iou = self.per_class_iou(hist)
        STR = ""
        for i in range(len(self.name)):
            STR = STR + self.name[i] + str(round(iou[i], 3)) + " "
        print(STR)
        miou = np.nanmean(iou)

        return round(acc, 3), round(acc_cls, 3), round(miou, 3)


class AverageMeter(object):
    def __init__(self):
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