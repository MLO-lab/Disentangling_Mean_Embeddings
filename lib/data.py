import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
# from tqdm import tqdm
import lib.notebook as notebook

tqdm = notebook.get_tqdm()

# transform for the inception v3 model
phi_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for PyTorch that wraps a tensor and provides access to its elements.

    Args:
        data_tensor (torch.Tensor): A tensor containing the dataset.

    Attributes:
        data (torch.Tensor): The tensor containing the dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the element at the specified index.
    """

    def __init__(self, data_tensor):
        """
        Initializes the CustomDataset with a tensor containing the data.

        Args:
            data_tensor (torch.Tensor): A tensor containing the dataset.
        """
        self.data = data_tensor

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of elements in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves the element at the specified index.

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            tuple: A tuple containing the element at the specified index twice.
        """
        return self.data[index], self.data[index]


def create_dataset(dataset_name, image_size, workers=8, subset=0, load_dataset_in_memory=False, type="train", flatten=False, use_phi=False, label=None):
    """
    Creates and returns a dataset and optionally pre-processes it for training or testing.

    Args:
        dataset_name (str): The name of the dataset to load. Supported values are "PathMNIST", "ChestMNIST", 
                            "OrganAMNIST", "OrganAMNISTReduced2", or any dataset in 'data/{dataset_name}/{type}/' directory.
        image_size (int or tuple): The size to which images will be resized.
        workers (int, optional): The number of worker threads to use for data loading. Default is 8.
        subset (int, optional): If non-zero, specifies the number of samples to include in the subset. Default is 0 (use entire dataset).
        load_dataset_in_memory (bool, optional): If True, loads the entire dataset into memory. Default is False.
        type (str, optional): The type of dataset split to use, typically "train" or "test". Default is "train".
        flatten (bool, optional): If True, flattens the images into vectors. Default is False.
        use_phi (bool, optional): If True, applies the inception_v3 transformation to the data and returns it. Default is False.
        label (int, optional): If specified, filters the dataset to include only samples with the given label. Default is None.

    Returns:
        dataset (torch.utils.data.Dataset): The dataset object.
        data_phi (np.ndarray, optional): If `use_phi` is True, returns the transformed data from the inception_v3 model; otherwise, returns None.

    Example:
        >>> dataset, data_phi = create_dataset("PathMNIST", image_size=(128, 128), workers=4, subset=1000, load_dataset_in_memory=True, type="train", flatten=True, use_phi=True)
    """
    data_phi = None
    if dataset_name == "PathMNIST":
        from medmnist import PathMNIST
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])
        dataset = PathMNIST(
            split=type,
            transform=data_transform,
            download=True
        )
    elif dataset_name == "ChestMNIST":
        from medmnist import ChestMNIST
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = ChestMNIST(
            split=type,
            transform=data_transform,
            download=True
        )
    elif dataset_name in ["OrganAMNIST", "OrganAMNISTReduced2"]:
        from medmnist import OrganAMNIST
        data_transform = transforms.Compose([transforms.ToTensor()])
        dataset = OrganAMNIST(
            split=type, 
            transform=data_transform, 
            download=True
        )
    else:
        dataset = dset.ImageFolder(
            root=f"data/{dataset_name}/{type}/",
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    if subset != 0:
        dataset = torch.utils.data.Subset(
            dataset,
            np.random.choice(
                len(dataset),
                subset,
                replace=False
            )
        )
    if load_dataset_in_memory:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True, num_workers=workers
        )
        data_list = []
        data_phi_list = []
        print("loading all images")
        for image in tqdm(dataloader):
            image = list(image)
            if label is not None:
                image[0] = image[0][image[1] == label]
                if len(image[0].size()) == 0:
                    continue
            if use_phi:
                data_phi_list.append(phi(image[0]))
            data_list.append(image[0])
        data = []
        data_phi = []
        print("stacking all images")
        for batch in tqdm(data_list):
            if flatten:
                data += list(batch.reshape(batch.shape[0], -1))
            else:
                data += list(batch)
        if use_phi:
            for batch in data_phi_list:
                data_phi += list(batch.detach().numpy())
            data_phi = np.array(data_phi)

        dataset = torch.stack(data)

        dataset = CustomDataset(dataset)

    if use_phi:
        return dataset, data_phi
    return dataset


def phi(tests):
    """
    Applies the Inception v3 model to a batch of images after transforming them.

    Args:
        tests (torch.Tensor or list): A batch of images to be processed. The input should be in a format compatible
                                      with the Inception v3 model, typically a 4D tensor with shape 
                                      (batch_size, channels, height, width).

    Returns:
        torch.Tensor: The output of the Inception v3 model after processing the input images.

    Example:
        >>> tests = torch.randn(8, 3, 299, 299)  # Example input tensor
        >>> outputs = phi(tests)
    """
    inception_model = models.inception_v3(pretrained=True)
    inception_model.eval()
    return inception_model(phi_transform(torch.tensor(tests)))


def main():
    create_dataset("ChestMNIST", 28, load_dataset_in_memory=True)


if __name__ == "__main__":
    main()
