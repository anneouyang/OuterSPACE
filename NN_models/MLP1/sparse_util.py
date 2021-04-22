import torch 

# sparsity level is defined as number of non-zero elements / total size

def get_sparsity(mat):
	zeros_count = (abs(mat - 0) <= 0.000001).sum()
	return (zeros_count, torch.numel(mat), 1 - zeros_count / torch.numel(mat))


def get_sparse_mask(mat, sparsity_level):
	threshold = torch.quantile(abs(mat), 1 - sparsity_level)
	mask = (mat > threshold)
	return mask


def prune_to_sparsity(mat, sparsity_level):
	# if already at or below desired sparsity level
	if get_sparsity(mat) <= sparsity_level:
		return mat
	mask = get_sparse_mask(mat, sparsity_level)
	return (mask, mat * mask)
