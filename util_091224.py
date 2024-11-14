import torch

def add_column_of_1s(
    matrix: torch.Tensor
) -> torch.Tensor:
    """
    Adds a column of 1s to a matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        A 2-dimensional tensor, that is a matrix.

    Returns
    -------
    A new tensor that is the matrix
    augmented by a column of 1s on the right.
    """
    matrix_aug = torch.concatenate(
        [
            matrix,
            torch.ones_like(matrix[:, :1])
        ],
        dim=1
    )

    return matrix_aug