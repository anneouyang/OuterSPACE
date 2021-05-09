import torch 

# sparsity level is defined as number of non-zero elements / total size

def get_sparsity(mat):
	zeros_count = (abs(mat - 0) <= 0.00000001).sum()
	return (zeros_count, torch.numel(mat), 1 - zeros_count / torch.numel(mat))

def get_prune_threshold(mat, sparsity_level):
	return torch.quantile(abs(mat), 1 - sparsity_level)

def get_sparse_mask(mat, sparsity_level):
	threshold = get_prune_threshold(mat, sparsity_level)
	mask = (mat > threshold)
	return mask

def prune_to_sparsity(mat, sparsity_level):
	# if already at or below desired sparsity level
	if get_sparsity(mat)[2] <= sparsity_level:
		return mat
	mask = get_sparse_mask(mat, sparsity_level)
	return mat * mask

def print_parameters_sparsity(model):
	# print(model.parameters)
	# print(model.named_parameters)
	print("parameters sparsity: ")
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, get_sparsity(param))