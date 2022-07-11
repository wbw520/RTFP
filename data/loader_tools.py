import data.transforms as joint_transforms
import torchvision.transforms as standard_transforms


def get_joint_transformations(args):
    aug_list = [
                joint_transforms.Resize(args),
                # joint_transforms.RandomSized(),
                joint_transforms.RandomRotate(10),
                joint_transforms.RandomCrop(args.crop_size),
                joint_transforms.RandomHorizontallyFlip(),
                ]
    return joint_transforms.Compose(aug_list)


def get_joint_transformations_val(args):
    aug_list = [
                joint_transforms.Resize(args),
                ]
    return joint_transforms.Compose(aug_list)


def get_standard_transformations():
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
