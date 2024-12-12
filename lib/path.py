import os
from lib.get_model import GANParameter


def create_path(path):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): The path to the directory that needs to be created.

    Notes:
        - If the directory already exists, this function does nothing.
        - Uses `os.makedirs` which can create intermediate directories if needed.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_training_path(model_name, dataset_name, max_images_per_epoch, params: list[GANParameter], model="GAN"):
    """
    Creates directories for saving trained models based on provided parameters.

    Args:
        model_name (str): The name of the model.
        dataset_name (str): The name of the dataset used for training.
        max_images_per_epoch (int): The maximum number of images per epoch.
        params (list of GANParameter): A list of `GANParameter` objects containing training parameters.
        model (str): The type of model, e.g., "GAN" or another model type. Default is "GAN".

    Notes:
        - Constructs the path based on the model type and parameters.
        - Calls `create_path` to ensure the directory structure exists.
    """
    for param in params:
        if model == "GAN":
            path = f"models/{model_name}/model_saves/" \
                   f"{dataset_name}-max_per_epoch{max_images_per_epoch}" \
                   f"-batch_size{param.batch_size}-ngf{param.ngf}-ndf{param.ndf}" \
                   f"-lr{param.lr}-beta1{param.beta1}/"
        elif model == "Diffusion":
            path = f"models/{model_name}/model_saves/" \
                   f"{dataset_name}-max_per_epoch{max_images_per_epoch}" \
                   f"-batch_size{param.batch_size}-size{param.size}" \
                   f"-lr{param.lr}-T{param.T}-deep_conv{param.deep_conv}/"
        else:
            raise ValueError(f"The model '{model}' does not exist. Use 'GAN' or 'Diffusion' instead.")
        create_path(path)


def create_evaluate_path(dataset_name, image_size):
    """
    Creates directories for saving evaluation plots.

    Args:
        dataset_name (str): The name of the dataset used for evaluation.
        image_size (int): The size of the images (assumed to be square).

    Notes:
        - Constructs paths for various types of evaluation plots.
        - Calls `create_path` to ensure the necessary directories are created.
    """
    create_path(f"plots/{dataset_name} {image_size}x{image_size}/var_pixel_wise/")
    create_path(f"plots/{dataset_name} {image_size}x{image_size}/corr_pixel_wise/")
    create_path(f"plots/{dataset_name} {image_size}x{image_size}/corr_pixel_wise_cluster/")
    create_path(f"plots/{dataset_name} {image_size}x{image_size}/image_wise/")
    create_path(f"plots/{dataset_name} {image_size}x{image_size}/cluster_wise/")
