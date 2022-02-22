from math import ceil
import torch
import numpy as np
import math


@torch.no_grad()
def inference_sliding(args, model, image, crop_size, stride_rate, classes):
    image_size = image.size()
    stride = int(math.ceil(crop_size[0] * stride_rate))
    tile_rows = ceil((image_size[2]-crop_size)/stride)
    tile_cols = ceil((image_size[3]-crop_size)/stride)
    b = image_size[0]

    full_probs = torch.from_numpy(np.zeros((b, classes, image_size[2], image_size[3]))).to(args.device)
    count_predictions = torch.from_numpy(np.zeros((b, classes, image_size[2], image_size[3]))).to(args.device)

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            if row == tile_rows - 1:
                y2 = image_size[2]
                y1 = image_size[2] - crop_size
            if col == tile_cols - 1:
                x2 = image_size[3]
                x1 = image_size[3] - crop_size

            img = image[:, :, y1:y2, x1:x2]

            with torch.set_grad_enabled(False):
                padded_prediction = model(img)
                if isinstance(padded_prediction, tuple):
                    padded_prediction = padded_prediction[0]
            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += padded_prediction  # accumulate the predictions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    _, preds = torch.max(full_probs, 1)
    return preds