import torch
import random


def set_random_seed(seed=None):
    """
    Sets a random seed for reproducibility in both Python's random module and PyTorch.

    Args:
        seed (int, optional): The seed value to set for randomness. If None, a random seed is generated. Default is None.

    Returns:
        int: The seed value that was set.

    Notes:
        - Sets the seed for Python's random number generator.
        - Sets the seed for PyTorch's random number generator.
        - Ensures that PyTorch operations use deterministic algorithms where possible, which is crucial for reproducibility.
        - The seed value is printed to the console for reference.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    print(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed
