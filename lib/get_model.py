class GANParameter:
    """
    A class to encapsulate parameters used in a GAN (Generative Adversarial Network).

    Attributes:
        batch_size (int): The batch size used for training.
        ngf (int): The number of generator filters in the first convolutional layer.
        ndf (int): The number of discriminator filters in the first convolutional layer.
        lr (float): The learning rate for training.
        beta1 (float): The beta1 parameter used in the optimizer.

    Methods:
        get(): Returns a tuple of the parameter values.
        __str__(): Returns a string representation of the parameter values.

    Example:
        >>> gan_params = GANParameter(batch_size=64, ngf=64, ndf=64, lr=0.0002, beta1=0.5)
        >>> print(gan_params)
        batch_size=64 ngf=64 ndf=64 lr=0.0002 beta1=0.5
        >>> params_tuple = gan_params.get()
        >>> print(params_tuple)
        (64, 64, 64, 0.0002, 0.5)
    """

    def __init__(self, batch_size, ngf, ndf, lr, beta1):
        """
        Initializes GANParameter with specified parameters.

        Args:
            batch_size (int): The batch size used for training.
            ngf (int): The number of generator filters in the first convolutional layer.
            ndf (int): The number of discriminator filters in the first convolutional layer.
            lr (float): The learning rate for training.
            beta1 (float): The beta1 parameter used in the optimizer.
        """
        self.batch_size = batch_size
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.beta1 = beta1

    def get(self):
        """
        Returns a tuple of the parameter values.

        Returns:
            tuple: A tuple containing (batch_size, ngf, ndf, lr, beta1).
        """
        return self.batch_size, self.ngf, self.ndf, self.lr, self.beta1

    def __str__(self):
        """
        Returns a string representation of the parameter values.

        Returns:
            str: A string representation of the GANParameter instance.
        """
        return f"batch_size={self.batch_size} ngf={self.ngf} ndf={self.ndf} lr={self.lr} beta1={self.beta1}"


class DiffusionParameter:
    """
    A class to encapsulate parameters used in diffusion processes.

    Attributes:
        batch_size (int): The batch size used in diffusion.
        size (int): The size parameter used in diffusion.
        lr (float): The learning rate used in diffusion.
        T (int): The T parameter used in diffusion.
        deep_conv (bool): Whether deep convolution is used in diffusion.

    Methods:
        get(): Returns a tuple of the parameter values.
        __str__(): Returns a string representation of the parameter values.

    Example:
        >>> diffusion_params = DiffusionParameter(batch_size=32, size=64, lr=0.001, T=10, deep_conv=True)
        >>> print(diffusion_params)
        batch_size=32 size=64 lr=0.001 T=10 deep_conv=True
        >>> params_tuple = diffusion_params.get()
        >>> print(params_tuple)
        (32, 64, 0.001, 10, True)
    """

    def __init__(self, batch_size, size, lr, T, deep_conv):
        """
        Initializes DiffusionParameter with specified parameters.

        Args:
            batch_size (int): The batch size used in diffusion.
            size (int): The size parameter used in diffusion.
            lr (float): The learning rate used in diffusion.
            T (int): The T parameter used in diffusion.
            deep_conv (bool): Whether deep convolution is used in diffusion.
        """
        self.batch_size = batch_size
        self.size = size
        self.lr = lr
        self.T = T
        self.deep_conv = deep_conv

    def get(self):
        """
        Returns a tuple of the parameter values.

        Returns:
            tuple: A tuple containing (batch_size, size, lr, T, deep_conv).
        """
        return self.batch_size, self.size, self.lr, self.T, self.deep_conv

    def __str__(self):
        """
        Returns a string representation of the parameter values.

        Returns:
            str: A string representation of the DiffusionParameter instance.
        """
        return f"batch_size={self.batch_size} size={self.size} lr={self.lr} T={self.T} deep_conv={self.deep_conv}"
    

class condDiffusionParameter:
    """
    A class to encapsulate parameters used in conditional diffusion processes.

    Attributes:
        batch_size (int): The batch size used in conditional diffusion.
        n_feat (int): The number of features used in conditional diffusion.
        lr (float): The learning rate used in conditional diffusion.
        T (int): The T parameter used in diffusion.
        n_classes (int): The number of classes for conditional diffusion.

    Methods:
        get(): Returns a tuple of the parameter values.
        __str__(): Returns a string representation of the parameter values.

    Example:
        >>> cond_diffusion_params = condDiffusionParameter(batch_size=32, n_feat=128, lr=0.001, T=10, n_classes=5)
        >>> print(cond_diffusion_params)
        batch_size=32 n_feat=128 lr=0.001 T=10 n_classes=5
        >>> params_tuple = cond_diffusion_params.get()
        >>> print(params_tuple)
        (32, 128, 0.001, 10, 5)
    """

    def __init__(self, batch_size, n_feat, lr, T, n_classes):
        """
        Initializes condDiffusionParameter with specified parameters.

        Args:
            batch_size (int): The batch size used in conditional diffusion.
            n_feat (int): The number of features used in conditional diffusion.
            lr (float): The learning rate used in conditional diffusion.
            T (int): The T parameter used in diffusion.
            n_classes (int): The number of classes for conditional diffusion.
        """
        self.batch_size = batch_size
        self.n_feat = n_feat
        self.lr = lr
        self.T = T
        self.n_classes = n_classes

    def get(self):
        """
        Returns a tuple of the parameter values.

        Returns:
            tuple: A tuple containing (batch_size, n_feat, lr, T, n_classes).
        """
        return self.batch_size, self.n_feat, self.lr, self.T, self.n_classes

    def __str__(self):
        """
        Returns a string representation of the parameter values.

        Returns:
            str: A string representation of the condDiffusionParameter instance.
        """
        return f"batch_size={self.batch_size} n_feat={self.n_feat} lr={self.lr} T={self.T} n_classes={self.n_classes}"


class GetGANModelParameters:
    """
    A class to manage a collection of GANParameter instances representing different GAN model configurations.

    Attributes:
        parameters (list): A list of GANParameter instances, each representing a different set of GAN model parameters.

    Methods:
        __len__(): Returns the number of GANParameter instances.
        __iter__(): Returns an iterator over the list of GANParameter instances.
        __getattr__(item): Retrieves a specific GANParameter instance by its index.
        get(indices=None): Retrieves the list of GANParameter instances, optionally filtered by indices.
        only(batch_size=None, ngf=None, ndf=None, lr=None, beta1=None): Filters and updates the list of parameters based on specified criteria.
        reset(): Resets the list of parameters to its original state.
        get_with(batch_size=None, ngf=None, ndf=None, lr=None, beta1=None): Retrieves GANParameter instances matching specified criteria.

    Example:
        >>> gan_params = GetGANModelParameters()
        >>> print(len(gan_params))
        7
        >>> for params in gan_params:
        >>>     print(params)
        batch_size=128 ngf=16 ndf=16 lr=0.0002 beta1=0.5
        batch_size=128 ngf=64 ndf=64 lr=0.0002 beta1=0.5
        batch_size=128 ngf=16 ndf=64 lr=0.0002 beta1=0.5
        batch_size=128 ngf=64 ndf=16 lr=0.0002 beta1=0.5
        batch_size=128 ngf=64 ndf=64 lr=0.0008 beta1=0.5
        batch_size=128 ngf=64 ndf=64 lr=1e-05 beta1=0.5
        only_64_64_lr_0008 = gan_params.only(ngf=64, ndf=64, lr=0.0008)
        >>> print(only_64_64_lr_0008)
        batch_size=128 ngf=64 ndf=64 lr=0.0008 beta1=0.5
    """

    _parameters = [
        GANParameter(*p)
        for p in [
            (128, 16, 16, 0.0002, 0.5),
            (128, 64, 64, 0.0002, 0.5),
            # (128, 32, 32, 0.0002, 0.5),
            (128, 16, 64, 0.0002, 0.5),
            (128, 64, 16, 0.0002, 0.5),
            # (32, 64, 64, 0.0002, 0.5),
            # (128, 64, 64, 0.0004, 0.5),
            # (128, 64, 64, 0.0001, 0.5),
            # (128, 64, 64, 0.0002, 0.25),
            # (128, 64, 64, 0.0002, 0.75),
            # (128, 64, 64, 0.002, 0.5),
            # (128, 64, 64, 0.02, 0.5),
            (128, 64, 64, 0.0008, 0.5),
            (128, 64, 64, 0.00001, 0.5),
        ]
    ]

    def __init__(self):
        self.parameters = self._parameters

    def __len__(self):
        """
        Returns the number of GANParameter instances available.

        Returns:
            int: Number of GANParameter instances.
        """
        return len(self.parameters)

    def __iter__(self):
        """
        Returns an iterator over the list of GANParameter instances.

        Returns:
            iterator: Iterator over GANParameter instances.
        """
        return iter(self.parameters)

    def __getattr__(self, item):
        """
        Retrieves a specific GANParameter instance by its index.

        Args:
            item (int): Index of the GANParameter instance.

        Returns:
            GANParameter: The GANParameter instance at the specified index.
        """
        return self.parameters[item]

    def get(self, indices=None):
        """
        Retrieves the list of GANParameter instances, optionally filtered by indices.

        Args:
            indices (list or None, optional): List of indices to retrieve. If None, returns all parameters. Default is None.

        Returns:
            list: List of GANParameter instances.
        """
        if indices is None:
            return self.parameters
        return [self.parameters[i] for i in indices]

    def only(self, batch_size=None, ngf=None, ndf=None, lr=None, beta1=None):
        """
        Filters and updates the list of parameters based on specified criteria.

        Args:
            batch_size (int or None, optional): Batch size criterion. Default is None.
            ngf (int or None, optional): Number of generator filters criterion. Default is None.
            ndf (int or None, optional): Number of discriminator filters criterion. Default is None.
            lr (float or None, optional): Learning rate criterion. Default is None.
            beta1 (float or None, optional): Beta1 criterion. Default is None.

        Returns:
            GetGANModelParameters: Updated instance of GetGANModelParameters with filtered parameters.
        """
        self.parameters = self.get_with(batch_size, ngf, ndf, lr, beta1)
        return self

    def reset(self):
        """
        Resets the list of parameters to its original state.

        Returns:
            GetGANModelParameters: Instance of GetGANModelParameters with original parameters.
        """
        self.parameters = self.get_with()
        return self

    def get_with(self, batch_size=None, ngf=None, ndf=None, lr=None, beta1=None):
        """
        Retrieves GANParameter instances matching specified criteria.

        Args:
            batch_size (int or None, optional): Batch size criterion. Default is None.
            ngf (int or None, optional): Number of generator filters criterion. Default is None.
            ndf (int or None, optional): Number of discriminator filters criterion. Default is None.
            lr (float or None, optional): Learning rate criterion. Default is None.
            beta1 (float or None, optional): Beta1 criterion. Default is None.

        Returns:
            list: List of GANParameter instances matching the specified criteria.
        """
        return [
            para for para in self._parameters
            if (
                    (batch_size is None or para.batch_size == batch_size) and
                    (ngf is None or para.ngf == ngf) and
                    (ndf is None or para.ndf == ndf) and
                    (lr is None or para.lr == lr) and
                    (beta1 is None or para.beta1 == beta1)
            )
        ]


class GetDiffusionModelParameters:
    """
    A class to manage a collection of DiffusionParameter instances representing different diffusion model configurations.

    Attributes:
        parameters (list): A list of DiffusionParameter instances, each representing a different set of diffusion model parameters.

    Methods:
        __len__(): Returns the number of DiffusionParameter instances.
        __iter__(): Returns an iterator over the list of DiffusionParameter instances.
        __getattr__(item): Retrieves a specific DiffusionParameter instance by its index.
        get(indices=None): Retrieves the list of DiffusionParameter instances, optionally filtered by indices.
        only(batch_size=None, size=None, lr=None, T=None, deep_conv=None): Filters and updates the list of parameters based on specified criteria.
        reset(): Resets the list of parameters to its original state.
        get_with(batch_size=None, size=None, lr=None, T=None, deep_conv=None): Retrieves DiffusionParameter instances matching specified criteria.

    Example:
        >>> diffusion_params = GetDiffusionModelParameters()
        >>> print(len(diffusion_params))
        6
        >>> for params in diffusion_params:
        >>>     print(params)
        batch_size=12 size=8 lr=0.0002 T=400 deep_conv=False
        batch_size=12 size=8 lr=0.001 T=400 deep_conv=False
        batch_size=12 size=8 lr=0.0002 T=1000 deep_conv=False
        batch_size=12 size=8 lr=2e-05 T=400 deep_conv=False
        batch_size=12 size=8 lr=2e-06 T=400 deep_conv=False
        batch_size=12 size=8 lr=2e-05 T=1000 deep_conv=False
        only_size_8_lr_0002 = diffusion_params.only(size=8, lr=0.0002)
        >>> print(only_size_8_lr_0002)
        batch_size=12 size=8 lr=0.0002 T=400 deep_conv=False
        batch_size=12 size=8 lr=0.0002 T=1000 deep_conv=False

    """

    _parameters = [
        DiffusionParameter(*p)
        for p in [
            (12, 8, 0.0002, 400, False),  # 0
            (12, 8, 0.001, 400, False),  # 1
            (12, 8, 0.0002, 1000, False),
            (12, 8, 0.00002, 400, False),
            (12, 8, 0.000002, 400, False),
            (12, 8, 0.00002, 1000, False),
        ]
    ]

    def __init__(self):
        self.parameters = self._parameters

    def __len__(self):
        """
        Returns the number of DiffusionParameter instances available.

        Returns:
            int: Number of DiffusionParameter instances.
        """
        return len(self.parameters)

    def __iter__(self):
        """
        Returns an iterator over the list of DiffusionParameter instances.

        Returns:
            iterator: Iterator over DiffusionParameter instances.
        """
        return iter(self.parameters)

    def __getattr__(self, item):
        """
        Retrieves a specific DiffusionParameter instance by its index.

        Args:
            item (int): Index of the DiffusionParameter instance.

        Returns:
            DiffusionParameter: The DiffusionParameter instance at the specified index.
        """
        return self.parameters[item]

    def get(self, indices=None):
        """
        Retrieves the list of DiffusionParameter instances, optionally filtered by indices.

        Args:
            indices (list or None, optional): List of indices to retrieve. If None, returns all parameters. Default is None.

        Returns:
            list: List of DiffusionParameter instances.
        """
        if indices is None:
            return self.parameters
        return [self.parameters[i] for i in indices]

    def only(self, batch_size=None, size=None, lr=None, T=None, deep_conv=None):
        """
        Filters and updates the list of parameters based on specified criteria.

        Args:
            batch_size (int or None, optional): Batch size criterion. Default is None.
            size (int or None, optional): Model size criterion. Default is None.
            lr (float or None, optional): Learning rate criterion. Default is None.
            T (int or None, optional): Time steps criterion. Default is None.
            deep_conv (bool or None, optional): Deep convolution criterion. Default is None.

        Returns:
            GetDiffusionModelParameters: Updated instance of GetDiffusionModelParameters with filtered parameters.
        """
        self.parameters = self.get_with(batch_size, size, lr, T, deep_conv)

    def reset(self):
        """
        Resets the list of parameters to its original state.

        Returns:
            GetDiffusionModelParameters: Instance of GetDiffusionModelParameters with original parameters.
        """
        self.parameters = self.get_with()

    def get_with(self, batch_size=None, size=None, lr=None, T=None, deep_conv=None):
        """
        Retrieves DiffusionParameter instances matching specified criteria.

        Args:
            batch_size (int or None, optional): Batch size criterion. Default is None.
            size (int or None, optional): Model size criterion. Default is None.
            lr (float or None, optional): Learning rate criterion. Default is None.
            T (int or None, optional): Time steps criterion. Default is None.
            deep_conv (bool or None, optional): Deep convolution criterion. Default is None.

        Returns:
            list: List of DiffusionParameter instances matching the specified criteria.
        """
        return [
            para for para in self._parameters
            if (
                    (batch_size is None or para.batch_size == batch_size) and
                    (size is None or para.size == size) and
                    (lr is None or para.lr == lr) and
                    (T is None or para.T == T) and
                    (deep_conv is None or para.deep_conv == deep_conv)
            )
        ]
    

class GetCondDiffusionModelParameters:
    """
    A class to manage a collection of condDiffusionParameter instances representing different conditional diffusion model configurations.

    Attributes:
        parameters (list): A list of condDiffusionParameter instances, each representing a different set of conditional diffusion model parameters.

    Methods:
        __len__(): Returns the number of condDiffusionParameter instances.
        __iter__(): Returns an iterator over the list of condDiffusionParameter instances.
        __getattr__(item): Retrieves a specific condDiffusionParameter instance by its index.
        get(indices=None): Retrieves the list of condDiffusionParameter instances, optionally filtered by indices.
        only(batch_size=None, n_feat=None, lr=None, T=None, n_classes=None): Filters and updates the list of parameters based on specified criteria.
        reset(): Resets the list of parameters to its original state.
        get_with(batch_size=None, n_feat=None, lr=None, T=None, n_classes=None): Retrieves condDiffusionParameter instances matching specified criteria.

    Example:
        >>> cond_diffusion_params = GetCondDiffusionModelParameters()
        >>> print(len(cond_diffusion_params))
        5
        >>> for params in cond_diffusion_params:
        >>>     print(params)
        batch_size=256 n_feat=128 lr=0.0001 T=400 n_classes=11
        batch_size=256 n_feat=128 lr=1e-05 T=400 n_classes=11
        batch_size=256 n_feat=128 lr=5e-06 T=400 n_classes=11
        batch_size=256 n_feat=128 lr=1e-06 T=400 n_classes=11
        batch_size=256 n_feat=128 lr=0.0002 T=1000 n_classes=11
        >>> selected_params = cond_diffusion_params.only(lr=1e-05)
        >>> for params in selected_params:
        >>>     print(params)
        batch_size=256 n_feat=128 lr=1e-05 T=400 n_classes=11
    """

    _parameters = [  # batch_size, n_feat, lr, T, n_classes
        condDiffusionParameter(*p)
        for p in [
            (256, 128, 1e-4, 400, 11),
            (256, 128, 1e-5, 400, 11),
            (256, 128, 5e-6, 400, 11),
            (256, 128, 1e-6, 400, 11),
            (256, 128, 0.0002, 1000, 11)
        ]
    ]

    def __init__(self):
        self.parameters = self._parameters

    def __len__(self):
        """
        Returns the number of condDiffusionParameter instances available.

        Returns:
            int: Number of condDiffusionParameter instances.
        """
        return len(self.parameters)

    def __iter__(self):
        """
        Returns an iterator over the list of condDiffusionParameter instances.

        Returns:
            iterator: Iterator over condDiffusionParameter instances.
        """
        return iter(self.parameters)

    def __getattr__(self, item):
        """
        Retrieves a specific condDiffusionParameter instance by its index.

        Args:
            item (int): Index of the condDiffusionParameter instance.

        Returns:
            condDiffusionParameter: The condDiffusionParameter instance at the specified index.
        """
        return self.parameters[item]

    def get(self, indices=None):
        """
        Retrieves the list of condDiffusionParameter instances, optionally filtered by indices.

        Args:
            indices (list or None, optional): List of indices to retrieve. If None, returns all parameters. Default is None.

        Returns:
            list: List of condDiffusionParameter instances.
        """
        if indices is None:
            return self.parameters
        return [self.parameters[i] for i in indices]

    def only(self, batch_size=None, n_feat=None, lr=None, T=None, n_classes=None):
        """
        Filters and updates the list of parameters based on specified criteria.

        Args:
            batch_size (int or None, optional): Batch size criterion. Default is None.
            n_feat (int or None, optional): Number of features criterion. Default is None.
            lr (float or None, optional): Learning rate criterion. Default is None.
            T (int or None, optional): Time steps criterion. Default is None.
            n_classes (int or None, optional): Number of classes criterion. Default is None.

        Returns:
            GetCondDiffusionModelParameters: Updated instance of GetCondDiffusionModelParameters with filtered parameters.
        """
        self.parameters = self.get_with(batch_size, n_feat, lr, T, n_classes)

    def reset(self):
        """
        Resets the list of parameters to its original state.

        Returns:
            GetCondDiffusionModelParameters: Instance of GetCondDiffusionModelParameters with original parameters.
        """
        self.parameters = self.get_with()

    def get_with(self, batch_size=None, n_feat=None, lr=None, T=None, n_classes=None):
        """
        Retrieves condDiffusionParameter instances matching specified criteria.

        Args:
            batch_size (int or None, optional): Batch size criterion. Default is None.
            n_feat (int or None, optional): Number of features criterion. Default is None.
            lr (float or None, optional): Learning rate criterion. Default is None.
            T (int or None, optional): Time steps criterion. Default is None.
            n_classes (int or None, optional): Number of classes criterion. Default is None.

        Returns:
            list: List of condDiffusionParameter instances matching the specified criteria.
        """
        return [
            para for para in self._parameters
            if (
                    (batch_size is None or para.batch_size == batch_size) and
                    (n_feat is None or para.n_feat == n_feat) and
                    (lr is None or para.lr == lr) and
                    (T is None or para.T == T) and
                    (n_classes is None or para.n_classes == n_classes)
            )
        ]


def get_model(model_name):
    """
    Imports and returns a specified model along with its image size.

    Args:
        model_name (str): The name of the model to import. Supported values are:
            - "DCGAN16x16"
            - "DCGAN28x28"
            - "DCGAN32x32"
            - "DCGAN64x64"
            - "Diffusion16x16"
            - "Diffusion28x28"
            - "Diffusion64x64"
            - "condDiffusion28x28"
            # - "smallDiffusion64x64" (commented out in the function)

    Returns:
        tuple: A tuple containing:
            - model (module): The imported model module.
            - image_size (int): The image size associated with the model.

    Raises:
        ValueError: If the provided model_name is not supported.

    Example:
        >>> model, image_size = get_model("DCGAN16x16")
        >>> print(model)
        <module 'models.DCGAN16x16.model' from '...'>
        >>> print(image_size)
        16
    """
    import models.DCGAN16x16.model as DCGAN16x16
    import models.DCGAN28x28.model as DCGAN28x28
    import models.DCGAN32x32.model as DCGAN32x32
    import models.DCGAN64x64.model as DCGAN64x64
    import models.Diffusion16x16.model as Diffusion16x16
    import models.Diffusion28x28.model as Diffusion28x28
    import models.Diffusion64x64.model as Diffusion64x64
    import models.condDiffusion28x28.model as condDiffusion28x28
    # import models.smallDiffusion64x64.model as smallDiffusion64x64
    if model_name == "DCGAN16x16":
        model = DCGAN16x16
        image_size = 16
    elif model_name == "DCGAN28x28":
        model = DCGAN28x28
        image_size = 28
    elif model_name == "DCGAN32x32":
        model = DCGAN32x32
        image_size = 32
    elif model_name == "DCGAN64x64":
        model = DCGAN64x64
        image_size = 64
    elif model_name == "Diffusion64x64":
        model = Diffusion64x64
        image_size = 64
    elif model_name == "Diffusion28x28":
        model = Diffusion28x28
        image_size = 28
    elif model_name == "Diffusion16x16":
        model = Diffusion16x16
        image_size = 16
    elif model_name == "condDiffusion28x28":
        model = condDiffusion28x28
        image_size = 28
    # elif model_name == "smallDiffusion64x64":
    #     model = smallDiffusion64x64
    #     image_size = 64
    else:
        raise ValueError
    return model, image_size
