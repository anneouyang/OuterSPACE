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
	if get_sparsity(mat)[2] <= sparsity_level:
		return mat
	mask = get_sparse_mask(mat, sparsity_level)
	return mat * mask


# only works for MLP1
def make_sparse_model_weights(model, sparsity_level):
	with torch.no_grad():
		model.fc1.weight = torch.nn.Parameter(prune_to_sparsity(model.fc1.weight.data, sparsity_level))
		model.fc1.bias = torch.nn.Parameter(prune_to_sparsity(model.fc1.bias.data, sparsity_level))
		model.fc2.weight = torch.nn.Parameter(prune_to_sparsity(model.fc2.weight.data, sparsity_level))
		model.fc2.bias = torch.nn.Parameter(prune_to_sparsity(model.fc2.bias.data, sparsity_level))
		model.fc3.weight = torch.nn.Parameter(prune_to_sparsity(model.fc3.weight.data, sparsity_level))
		model.fc3.bias = torch.nn.Parameter(prune_to_sparsity(model.fc3.bias.data, sparsity_level))
		pass

def print_parameters_sparsity(model):
	for param in model.parameters():
		print(get_sparsity(param))