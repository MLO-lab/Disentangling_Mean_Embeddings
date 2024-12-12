import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_generator(model, nc, nz, ngf, device=torch.device("cpu"), ngpu=0, print_model=False):
    """
    Creates and initializes a generator model for GANs (Generative Adversarial Networks).

    Args:
        model (module): The model module containing the Generator class.
        nc (int): The number of output channels.
        nz (int): The size of the latent vector (input to the generator).
        ngf (int): The number of generator filters in the first convolutional layer.
        device (torch.device, optional): The device to run the model on (CPU or CUDA). Default is CPU.
        ngpu (int, optional): The number of GPUs to use. Default is 0 (use CPU).
        print_model (bool, optional): If True, prints the model architecture. Default is False.

    Returns:
        netG (torch.nn.Module): The initialized generator model.

    Example:
        >>> model_module = your_model_module
        >>> netG = create_generator(
        ...     model_module, 
        ...     nc=3, 
        ...     nz=100, 
        ...     ngf=64, 
        ...     device=torch.device("cuda"), 
        ...     ngpu=1, 
        ...     print_model=True
        ... )
    """
    # Create the generator
    netG = model.Generator(ngpu, nc, nz, ngf).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    if print_model:
        print(netG)
    return netG


def create_discriminator(model, nc, ndf, device=torch.device("cpu"), ngpu=0, print_model=False):
    """
    Creates and initializes a discriminator model for GANs (Generative Adversarial Networks).

    Args:
        model (module): The model module containing the Discriminator class.
        nc (int): The number of input channels.
        ndf (int): The number of discriminator filters in the first convolutional layer.
        device (torch.device, optional): The device to run the model on (CPU or CUDA). Default is CPU.
        ngpu (int, optional): The number of GPUs to use. Default is 0 (use CPU).
        print_model (bool, optional): If True, prints the model architecture. Default is False.

    Returns:
        netD (torch.nn.Module): The initialized discriminator model.

    Example:
        >>> model_module = your_model_module
        >>> netD = create_discriminator(
        ...     model_module, 
        ...     nc=3, 
        ...     ndf=64, 
        ...     device=torch.device("cuda"), 
        ...     ngpu=1, 
        ...     print_model=True
        ... )
    """
    # Create the Discriminator
    netD = model.Discriminator(ngpu, nc, ndf).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    if print_model:
        print(netD)
    return netD


def import_model(model, model_name, dataset_name, max_images_per_epoch, nc, nz, model_nr, epoch, batch_size, ngf, ndf, lr, beta1, device=torch.device("cpu"), ngpu=0):
    """
    Imports and initializes a pre-trained generator model from a saved checkpoint file.

    Args:
        model (module): The model module containing the Generator class.
        model_name (str): The name of the model for directory structure.
        dataset_name (str): The name of the dataset for directory structure.
        max_images_per_epoch (int): Maximum number of images per epoch used during training.
        nc (int): The number of input channels.
        nz (int): The size of the latent vector (input to the generator).
        model_nr (int): The model number or identifier.
        epoch (int): The epoch number of the saved model checkpoint to load.
        batch_size (int): The batch size used during training.
        ngf (int): The number of generator filters in the first convolutional layer.
        ndf (int): The number of discriminator filters in the first convolutional layer.
        lr (float): The learning rate used during training.
        beta1 (float): The beta1 parameter used during training.
        device (torch.device, optional): The device to run the model on (CPU or CUDA). Default is CPU.
        ngpu (int, optional): The number of GPUs to use. Default is 0 (use CPU).

    Returns:
        netG (torch.nn.Module): The imported and initialized generator model.

    Example:
        >>> model_module = your_model_module
        >>> netG = import_model(
        ...     model_module, 
        ...     model_name="generator_model", 
        ...     dataset_name="CelebA", 
        ...     max_images_per_epoch=5000, 
        ...     nc=3, 
        ...     nz=100, 
        ...     model_nr=1, 
        ...     epoch=50, 
        ...     batch_size=64, 
        ...     ngf=64, 
        ...     ndf=64, 
        ...     lr=0.0002, 
        ...     beta1=0.5, 
        ...     device=torch.device("cuda"), 
        ...     ngpu=1
        ... )
    """
    netG = model.Generator(ngpu, nc, nz, ngf).to(device)
    netG.load_state_dict(
        torch.load(
            f"models/{model_name}/model_saves/{dataset_name}-"
            f"max_per_epoch{max_images_per_epoch}-batch_size{batch_size}-"
            f"ngf{ngf}-ndf{ndf}-lr{lr}-beta1{beta1}/generator-model{model_nr}-"
            f"epoch{epoch}.pkl")
    )
    netG.eval()
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    return netG


def get_images_from_model(model, model_name, dataset_name, max_images_per_epoch, nc, nz, params, noise, epoch=1, n=10, device=torch.device("cpu"), ngpu=0, model_nrs=None):
    """
    Generates images using multiple instances of a pre-trained generator model.

    Args:
        model (module): The model module containing the generator and import_model functions.
        model_name (str): The name of the model for directory structure.
        dataset_name (str): The name of the dataset for directory structure.
        max_images_per_epoch (int): Maximum number of images per epoch.
        nc (int): The number of input channels.
        nz (int): The size of the latent vector (input to the generator).
        params (tuple or GANParameter instance): A tuple or a GANParameter instance containing additional parameters specific to the generator model.
        noise (torch.Tensor): The noise tensor to be used as input for generating images.
        epoch (int, optional): The epoch number of the saved model to load. Default is 1.
        n (int, optional): The number of different model instances to sample from. Default is 10.
        device (torch.device, optional): The device to run the model on (CPU or CUDA). Default is CPU.
        ngpu (int, optional): The number of GPUs to use. Default is 0 (use CPU).
        model_nrs (list or None, optional): A list of specific model numbers to use for generation. 
                                           If None, generates images from a range of `n` models. Default is None.

    Returns:
        np.ndarray: An array of generated images.

    Example:
        >>> model_module = your_model_module
        >>> noise = torch.randn(16, 100)  # Example noise tensor shape
        >>> params = (64, 64, 0.0002, 0.5)  # Example parameters for the generator
        >>> images = get_images_from_model(
        ...     model_module, 
        ...     model_name="generator_model", 
        ...     dataset_name="CelebA", 
        ...     max_images_per_epoch=5000, 
        ...     nc=3, 
        ...     nz=100, 
        ...     params=params, 
        ...     noise=noise, 
        ...     epoch=50, 
        ...     n=10, 
        ...     device=torch.device("cuda"), 
        ...     ngpu=1, 
        ...     model_nrs=[1, 3, 5]  # Example specific model numbers
        ... )
    """
    images = []
    generator = range(n) if model_nrs is None else model_nrs
    for model_nr in generator:
        temp_model = import_model(model, model_name, dataset_name, max_images_per_epoch, nc, nz, model_nr, epoch, device=device, ngpu=ngpu, *params)
        image = temp_model(noise[:params[0]])
        images.append(image)
        del temp_model
    return torch.stack(images).cpu().detach().numpy()
