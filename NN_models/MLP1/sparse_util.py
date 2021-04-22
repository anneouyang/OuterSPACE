def get_sparsity(mat):
	pass


def prune_to_sparsity(mat, sparsity_level):
	# if already at or below desired sparsity level
	if get_sparsity(mat) <= sparsity_level:
		return mat

