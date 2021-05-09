import torch
import torch.nn as nn

import argparse
import copy
import os
import sys

from models import MLP1
from dataloaders import dataloaders
from sparse_util import *

import pickle
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("Could not find GPU! Using CPU only")

def save_model_weights(model, save_dir, name='best_weights.pt'):
	best_model_wts = copy.deepcopy(model.state_dict())
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)
	torch.save(best_model_wts, os.path.join(save_dir, name))


def load_model_weights(model, save_dir, name='best_weights.pt'):
	model.load_state_dict(torch.load(os.path.join(save_dir, name)))


def save_training_stats(train_losses, train_accs, val_losses, val_accs, save_dir):
	with open(save_dir + "/train_stats", "wb") as f:
		pickle.dump((train_losses, train_accs, val_losses, val_accs), f)
	f.close()


def load_training_stats(save_dir):
	with open(save_dir + "/train_stats", "rb") as f:
		a = pickle.load(f)
	f.close()
	return a


def plot_training_stats(train_losses, train_accs, val_losses, val_accs, save_dir):

	plt.figure()
	plt.title("Training and Validation Loss")
	plt.plot(val_losses,label="val")
	plt.plot(train_losses,label="train")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(save_dir + "/loss_plot.png")

	plt.figure()
	plt.title("Training and Validation Accuracy")
	plt.plot(val_accs,label="val")
	plt.plot(train_accs,label="train")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.savefig(save_dir + "/acc_plot.png")


def train(model, num_epochs=10, save_dir=None):

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()
	lambda1, lambda2, lambda3 = 0.1, 0.01, 0.01

	train_losses = []
	train_accs = []
	val_losses = []
	val_accs = []

	for epoch in range(num_epochs):
		print("", flush=True)
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))

		for phase in ['val', 'train']:
			if phase == 'train':
				model.train() 
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			print_act_sparsity = True

			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs, print_act_sparsity)
					print_act_sparsity = False
					loss = criterion(outputs, labels)
					all_fc1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
					all_fc2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])
					all_fc3_params = torch.cat([x.view(-1) for x in model.fc3.parameters()])
					l2_regularization_fc1 = lambda1 * torch.norm(all_fc1_params, 2)
					l2_regularization_fc2 = lambda2 * torch.norm(all_fc2_params, 2)
					l2_regularization_fc3 = lambda3 * torch.norm(all_fc3_params, 2)
					loss = loss + (l2_regularization_fc1 + l2_regularization_fc2 + l2_regularization_fc3)
					_, preds = torch.max(outputs, 1)
					if phase == 'train':
						loss.backward()
						for k, m in enumerate(model.modules()):
							if isinstance(m, nn.Linear):
								weight_copy = m.weight.data.abs().clone()
								mask = weight_copy.gt(0).float()
								m.weight.grad.data.mul_(mask)
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			if phase == 'val' and save_dir != None and (len(val_accs) == 0 or epoch_acc > max(val_accs)):
				save_model_weights(model, save_dir, "weights_epoch_" + str(epoch) + ".pt")
				save_model_weights(model, save_dir, "best_weights.pt")

			if phase == 'train':
				train_losses.append(float(epoch_loss))
				train_accs.append(float(epoch_acc))
			else:
				val_losses.append(float(epoch_loss))
				val_accs.append(float(epoch_acc))

			print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			print("", flush=True)

		print_parameters_sparsity(model=model)

	if save_dir != None:
		save_model_weights(model, save_dir, "weights_epoch_" + str(num_epochs - 1) + ".pt")
		save_training_stats(train_losses, train_accs, val_losses, val_accs, save_dir)
		plot_training_stats(train_losses, train_accs, val_losses, val_accs, save_dir)


def eval(model, save_dir=None, name="best_weights.pt"):

	model.sparsity_level = None

	if save_dir != None:
		load_model_weights(model, save_dir, name)

	print("")
	print("evaluating using model saved at", save_dir+"/"+name, "\n")

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
			outputs = model(inputs, print_activation_sparsity=True)
			loss = criterion(outputs, labels)
			_, preds = torch.max(outputs, 1)

		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(preds == labels.data)

	total_loss = running_loss / len(test_loader.dataset)
	total_acc = running_corrects.double() / len(test_loader.dataset)

	print("", flush=True)
	print('test loss: {:.4f} acc: {:.4f}'.format(total_loss, total_acc))
	print("", flush=True)

	print_parameters_sparsity(model=model)


def main():

	parser = argparse.ArgumentParser(description="MLP 1")
	parser.add_argument("num_epochs", help="Number of epochs to train", type=int, default=-1, nargs="?")
	parser.add_argument("saved_model_name", help="Name of saved model", type=str, default="no", nargs="?")
	parser.add_argument("eval_weights", help="Name of weights used in eval", type=str, default="best_weights.pt", nargs="?")

	args = parser.parse_args()
	num_epochs, saved_model_name, eval_weights = args.num_epochs, args.saved_model_name, args.eval_weights

	if saved_model_name == "no":
		save_dir = None
	else:
		save_dir = 'saved_weights/' + str(saved_model_name)
		try:
			os.stat(save_dir)
		except:
			os.mkdir(save_dir)

	model = MLP1(sparsity_level=0.1)
	load_model_weights(model, 'saved_weights/no_prune')

	if num_epochs > 0:
		train(model=model, num_epochs=num_epochs, save_dir=save_dir)
	else:
		print("skipped training")
	load_training_stats(save_dir)
	eval(model=model, save_dir=save_dir, name=eval_weights)
	# print_parameters_sparsity(model=model)

	# a = load_training_stats(save_dir)
	# plot_training_stats(a[0], a[1], a[2], a[3], save_dir)


if __name__ == '__main__':
	main()