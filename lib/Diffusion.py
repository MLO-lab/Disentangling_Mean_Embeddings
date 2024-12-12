import torch
import torch.nn as nn


def create_unet(model, nc=3, image_size=64, deep_conv=False, size=8, device=torch.device("cpu"), ngpu=0):
    """
    Creates a U-Net model and initializes it for training or inference.

    Args:
        model (module): The model module containing the U-Net implementation.
        nc (int, optional): The number of input channels. Default is 3.
        image_size (int, optional): The size of the input images (assumed to be square). Default is 64.
        deep_conv (bool, optional): Whether to use deep convolutions in the U-Net. Default is False.
        size (int, optional): A parameter to control the size/depth of the U-Net. Default is 8.
        device (torch.device, optional): The device to run the model on (CPU or CUDA). Default is CPU.
        ngpu (int, optional): The number of GPUs to use. Default is 0 (use CPU).

    Returns:
        net (torch.nn.Module): The initialized U-Net model.

    Example:
        >>> unet_model = create_unet(model_module, nc=1, image_size=128, deep_conv=True, size=16, device=torch.device("cuda"), ngpu=2)
    """
    net = model.UNet(nc, image_size, deep_conv, size, device=device).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))

    return net


def import_model(model, model_name, dataset_name, max_images_per_epoch, nc, image_size, model_nr, epoch, batch_size, size, lr, T, deep_conv, device=torch.device("cpu"), ngpu=0):
    """
    Imports a pre-trained U-Net model from a saved state dictionary.

    Args:
        model (module): The model module containing the U-Net implementation.
        model_name (str): The name of the model for directory structure.
        dataset_name (str): The name of the dataset for directory structure.
        max_images_per_epoch (int): Maximum number of images per epoch.
        nc (int): The number of input channels.
        image_size (int): The size of the input images (assumed to be square).
        model_nr (int): The model number to differentiate between multiple saved models.
        epoch (int): The epoch number of the saved model.
        batch_size (int): The batch size used during training.
        size (int): A parameter to control the size/depth of the U-Net.
        lr (float): The learning rate used during training.
        T (float): A temperature parameter used during training.
        deep_conv (bool): Whether to use deep convolutions in the U-Net.
        device (torch.device, optional): The device to run the model on (CPU or CUDA). Default is CPU.
        ngpu (int, optional): The number of GPUs to use. Default is 0 (use CPU).

    Returns:
        net (torch.nn.Module): The U-Net model loaded with the pre-trained weights.

    Example:
        >>> model_module = your_model_module
        >>> net = import_model(
        ...     model_module, 
        ...     model_name="unet_model", 
        ...     dataset_name="PathMNIST", 
        ...     max_images_per_epoch=5000, 
        ...     nc=3, 
        ...     image_size=64, 
        ...     model_nr=1, 
        ...     epoch=50, 
        ...     batch_size=16, 
        ...     size=8, 
        ...     lr=0.001, 
        ...     T=1.0, 
        ...     deep_conv=True, 
        ...     device=torch.device("cuda"), 
        ...     ngpu=2
        ... )
    """
    net = model.UNet(nc=nc, image_size=image_size, deep_conv=deep_conv, size=size, device=device).to(device)
    net.load_state_dict(
        torch.load(
            f"models/{model_name}/model_saves/"
            f"{dataset_name}-max_per_epoch{max_images_per_epoch}"
            f"-batch_size{batch_size}-size{size}"
            f"-lr{lr}-T{T}-deep_conv{deep_conv}/"
            f"unet-model{model_nr}-epoch{epoch}.pkl")
    )
    net.eval()
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    return net


def get_images_from_model(model, model_name, dataset_name, max_images_per_epoch, nc, image_size, params, noise, intern_noise=None, epoch=1, n=10, device=torch.device("cpu"), ngpu=0):
    """
    Generates images using a diffusion model by sampling from multiple instances of a pre-trained U-Net model.

    Args:
        model (module): The model module containing the U-Net and Diffusion implementations.
        model_name (str): The name of the model for directory structure.
        dataset_name (str): The name of the dataset for directory structure.
        max_images_per_epoch (int): Maximum number of images per epoch.
        nc (int): The number of input channels.
        image_size (int): The size of the input images (assumed to be square).
        params (tuple or DiffusionParameter): A tuple  or a DiffusionParameter containing (batch_size, size, lr, T, deep_conv).
        noise (torch.Tensor): The noise tensor to be used for sampling.
        intern_noise (torch.Tensor, optional): Additional internal noise tensor for sampling with shape (T, *noise.shape). Default is None.
        epoch (int, optional): The epoch number of the saved model to load. Default is 1.
        n (int, optional): The number of different model instances to sample from. Default is 10.
        device (torch.device, optional): The device to run the model on (CPU or CUDA). Default is CPU.
        ngpu (int, optional): The number of GPUs to use. Default is 0 (use CPU).

    Returns:
        np.ndarray: An array of generated images.

    Example:
        >>> model_module = your_model_module
        >>> noise = torch.randn(1, 3, 64, 64)
        >>> params = (16, 8, 0.001, 1000, True)
        >>> images = get_images_from_model(
        ...     model_module, 
        ...     model_name="ChestMNIST 64x64", 
        ...     dataset_name="ChestMNIST", 
        ...     max_images_per_epoch=10000, 
        ...     nc=1, 
        ...     image_size=64, 
        ...     params=params, 
        ...     noise=noise, 
        ...     intern_noise=None, 
        ...     epoch=50, 
        ...     n=10, 
        ...     device=torch.device("cuda"), 
        ...     ngpu=2
        ... )
    """
    images = []
    for model_nr in range(n):
        batch_size, size, lr, T, deep_conv = params
        dif = model.Diffusion(T=T, image_size=image_size, nc=nc, device=device)
        temp_model = import_model(model, model_name, dataset_name, max_images_per_epoch, nc, image_size, model_nr, epoch, *params, device=device, ngpu=ngpu)
        image = dif.sample(temp_model, x=noise, intern_noise=intern_noise, prt=False)
        images.append(image)
        del temp_model
    return torch.stack(images).cpu().detach().numpy()


def get_images_from_cond_model(model, model_name, dataset_name, max_images_per_epoch, nc, image_size, params, noise, intern_noise, class_id, epoch=1, n=10, device=torch.device("cpu"), ngpu=0, ws_test=None, betas=(1e-4, 0.02)):
    """
    Generates images using a conditional model by sampling from multiple instances of a pre-trained model.

    Args:
        model (module): The model module containing the conditional model and its evaluation function.
        model_name (str): The name of the model for directory structure.
        dataset_name (str): The name of the dataset for directory structure.
        max_images_per_epoch (int): Maximum number of images per epoch.
        nc (int): The number of input channels.
        image_size (int): The size of the input images (assumed to be square).
        params (tuple or condDiffusionParameter instance): A tuple or a condDiffusionParameter instance containing (batch_size, n_feat, lr, T, n_classes).
        noise (torch.Tensor): The noise tensor to be used for sampling.
        intern_noise (torch.Tensor): Additional internal noise tensor for sampling with shape (T, *noise.shape).
        class_id (int): The class ID for conditional generation.
        epoch (int, optional): The epoch number of the saved model to load. Default is 1.
        n (int, optional): The number of different model instances to sample from. Default is 10.
        device (torch.device, optional): The device to run the model on (CPU or CUDA). Default is CPU.
        ngpu (int, optional): The number of GPUs to use. Default is 0 (use CPU).
        ws_test (optional): Additional parameters for the model evaluation. Default is None.
        betas (tuple, optional): The beta parameters for the diffusion process. Default is (1e-4, 0.02).

    Returns:
        list: A list of generated images as torch.Tensors.

    Example:
        >>> model_module = your_model_module
        >>> noise = torch.randn(1, 3, 64, 64)
        >>> intern_noise = torch.randn(1000, 1, 3, 64, 64)  # Example shape for intern_noise
        >>> params = (16, 128, 0.001, 1000, 10)
        >>> class_id = 5
        >>> images = get_images_from_cond_model(
        ...     model_module, 
        ...     model_name="conditional_model", 
        ...     dataset_name="ChestMNIST", 
        ...     max_images_per_epoch=10000, 
        ...     nc=3, 
        ...     image_size=64, 
        ...     params=params, 
        ...     noise=noise, 
        ...     intern_noise=intern_noise, 
        ...     class_id=class_id, 
        ...     epoch=50, 
        ...     n=10, 
        ...     device=torch.device("cuda"), 
        ...     ngpu=2, 
        ...     ws_test=None, 
        ...     betas=(1e-4, 0.02)
        ... )
    """
    images = []  
    # def eval(x, class_id, n_T, device, n_classes, save_path, n_feat=128, *, in_channels, betas, ws_test):
    for model_nr in range(n):
        batch_size, n_feat, lr, T, n_classes = params
        save_path = f"models/{model_name}/data/{dataset_name}-max_per_epoch{max_images_per_epoch}-batch_size{batch_size}-lr{lr}-T{T}/unet-model{model_nr}-epoch{epoch}.pkl"
        image = model.eval(x=noise, class_id=class_id, n_T=T, intern_noise=intern_noise, device=device, n_classes=n_classes, save_path=save_path, n_feat=128, in_channels=nc, betas=betas, ws_test=ws_test).cpu()
        images.append(image)
    return images
