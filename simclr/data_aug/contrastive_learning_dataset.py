from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    @staticmethod
    def get_oct_simclr_pipeline_transform():
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        normalize = [(0.12, 0.12, 0.12), (0.19, 0.19, 0.19)]
        data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(*normalize),
            ]
        )
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(32), n_views),
                download=True,
            ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(96), n_views),
                download=True,
            ),
            "oct": lambda: datasets.ImageFolder(
                self.root_folder, transform=ContrastiveLearningViewGenerator(self.get_oct_simclr_pipeline_transform(), n_views)
            ),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
