import torch.nn as nn
import torch.nn.functional as F

from sparse_util import *

class MLP1(nn.Module):

	def __init__(self):
		super(MLP1, self).__init__()
		hidden1 = 1000
		hidden2 = 1000
		self.fc1 = nn.Linear(28 * 28, hidden1)
		self.fc2 = nn.Linear(hidden1, hidden2)
		self.fc3 = nn.Linear(hidden2, 10)
	
	def forward(self, x, print_activation_sparsity=False):

		x = x.view(-1, 28 * 28)

		# if self.sparsity_level != None:
		# 	with torch.no_grad():
		# 		self.fc1.weight.mul_(get_sparse_mask(self.fc1.weight.data, self.sparsity_level))
		# 		# self.fc1.bias.mul_(get_sparse_mask(self.fc1.bias.data, self.sparsity_level))
		# 		self.fc2.weight.mul_(get_sparse_mask(self.fc2.weight.data, self.sparsity_level))
		# 		# self.fc2.bias.mul_(get_sparse_mask(self.fc2.bias.data, self.sparsity_level))
		# 		self.fc3.weight.mul_(get_sparse_mask(self.fc3.weight.data, self.sparsity_level))
		# 		# self.fc3.bias.mul_(get_sparse_mask(self.fc3.bias.data, self.sparsity_level))
		
		x1 = F.relu(self.fc1(x))
		if print_activation_sparsity:
			print("activation sparsity: ")
			print("after fc1", get_sparsity(x1))
		
		x2 = F.relu(self.fc2(x1))
		if print_activation_sparsity:
			print("after fc2", get_sparsity(x2), flush=True)
		
		x3 = self.fc3(x2)

		return (x3, (x1, x2))