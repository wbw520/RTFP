from model.segmenter import create_segmenter
from model.psp_net import PSPNet


def model_generation(args):
    if args.model_name == "PSPNet":
        return PSPNet(num_classes=args.num_classes)
    elif args.model_name == "Segmenter":
        return create_segmenter(args)