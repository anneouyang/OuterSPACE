import torch.nn as nn
import torch.nn.functional as F

class MLP1(nn.Module):

	def __init__(self):
		super(MLP1, self).__init__()
		hidden1 = 1024
		hidden2 = 1024
		self.fc1 = nn.Linear(28 * 28, hidden1)
		self.fc2 = nn.Linear(hidden1, hidden2)
		self.fc3 = nn.Linear(hidden2, 10)
		self.dropout = nn.Dropout(0.2)
	
	def forward(self, x):
		x = x.view(-1, 28 * 28)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		x = self.dropout(x)
		x = self.fc3(x)
		return x