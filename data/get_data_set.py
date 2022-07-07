from data import cityscapes
from data import facade
from data.loader_tools import get_joint_transformations, get_standard_transformations, get_joint_transformations_val


def get_data(args):
    joint_transformations = get_joint_transformations(args)
    joint_transformations_val = get_joint_transformations_val(args)
    standard_transformations = get_standard_transformations()

    if args.dataset == "cityscapes":
        train_set = cityscapes.CityScapes(args, 'fine', 'train', joint_transform=joint_transformations,
                                          standard_transform=standard_transformations)
        val_set = cityscapes.CityScapes(args, 'fine', 'val', joint_transform=None,
                                        standard_transform=standard_transformations)
        ignore_index = cityscapes.ignore_label
        args.num_classes = cityscapes.num_classes
    elif args.dataset == "facade":
        train_set = facade.Facade(args, 'train', joint_transform=joint_transformations,
                                              standard_transform=standard_transformations)
        val_set = facade.Facade(args, 'val', joint_transform=joint_transformations_val,
                                            standard_transform=standard_transformations)
        ignore_index = facade.ignore_label
        args.num_classes = facade.num_classes
    else:
        raise "dataset name error !"

    return train_set, val_set, ignore_index