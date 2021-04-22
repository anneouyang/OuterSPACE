import torch
import torch.nn as nn

import copy
import os

from models import MLP1
from dataloaders import dataloaders
from sparse_util import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")


def save_model_weights(model, save_dir, name='weights.pt'):
	best_model_wts = copy.deepcopy(model.state_dict())
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)
	torch.save(best_model_wts, os.path.join(save_dir, name))


def load_model_weights(model, save_dir, name='weights.pt'):
	model.load_state_dict(torch.load(os.path.join(save_dir, name)))


def train(model, num_epochs=10, save_dir=None):

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		# make_sparse_model_weights(model, 0.5)

		# for param in model.parameters():
		# 	# print(get_sparsity(param))
		# 	print(param.size())

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train() 
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					loss = criterion(outputs, labels)
					_, preds = torch.max(outputs, 1)
					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

	if save_dir != None:
		save_model_weights(model, save_dir)


def eval(model, save_dir=None):

	if save_dir != None:
		load_model_weights(model, save_dir)

	model.eval()

	test_loader = dataloaders['test']
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	running_loss = 0.0
	running_corrects = 0

	for inputs, labels in test_loader:
		inputs = inputs.to(device)
		labels = labels.to(device)

		with torch.set_grad_enabled(False):
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			_, preds = torch.max(outputs, 1)

		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(preds == labels.data)

	total_loss = running_loss / len(test_loader.dataset)
	total_acc = running_corrects.double() / len(test_loader.dataset)

	print('test Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))


def main():
	save_dir = 'saved_weights/test1/'
	model = MLP1()
	train(model=model, num_epochs=3, save_dir=None)
	eval(model=model, save_dir=None)
	make_sparse_model_weights(model, 0.5)
	eval(model=model, save_dir=None)

if __name__ == '__main__':
	main()