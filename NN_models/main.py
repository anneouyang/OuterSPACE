import torch
import torch.nn as nn

import argparse
import copy
import os
import sys
import copy

from models import *
from dataloaders import dataloaders
from sparse_util import *
from util import *

import pickle
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("Could not find GPU! Using CPU only")


def eval(model, load_dir=None, name="best_weights.pt"):

	model.sparsity_level = None

	if load_dir != None:
		load_model_weights(model, load_dir, name)
		print("")
		print("evaluating using model saved at", load_dir+"/"+name, "\n")

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
			outputs = model(inputs, print_activation_sparsity=True)[0]
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



def train(model, num_epochs=10, save_dir=None, l2reg=False, finetune=False, model_type="MLP1"):

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	if l2reg:
		if model_type == "MLP1":
			lambda1, lambda2, lambda3, alambda1, alambda2 = 0.01, 0.01, 0.01, 0.002, 0.002
		elif model_type == "LeNet":
			lambda1, lambda2, lambda3 = 0.01, 0.01, 0
			alambdaf0, alambdaf1, alambdaf2 = 0, 0, 0
			alambdac1, alambdac2 = 0.001, 0.001
	else:
		if model_type == "MLP1":
			lambda1, lambda2, lambda3, alambda1, alambda2 = 0, 0, 0, 0, 0
		elif model_type == "LeNet":
			lambda1, lambda2, lambda3, alambdaf0, alambdaf1, alambdaf2, alambdac1, alambdac2 = 0, 0, 0, 0, 0, 0, 0, 0

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
					outputs, activations = model(inputs, print_act_sparsity)
					print_act_sparsity = False
					loss = criterion(outputs, labels)
					if l2reg:
						# print("l2reg")
						if model_type == "MLP1":
							# print("l2reg")
							all_fc1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
							all_fc2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])
							all_fc3_params = torch.cat([x.view(-1) for x in model.fc3.parameters()])
							l2_regularization_fc1 = lambda1 * torch.norm(all_fc1_params, 2)
							l2_regularization_fc2 = lambda2 * torch.norm(all_fc2_params, 2)
							l2_regularization_fc3 = lambda3 * torch.norm(all_fc3_params, 2)
							l2_regularization_act1 = alambda1 * torch.norm(activations[0], 2)
							l2_regularization_act2 = alambda2 * torch.norm(activations[1], 2)
							loss = loss + (l2_regularization_fc1 + l2_regularization_fc2 + l2_regularization_fc3 + l2_regularization_act1 + l2_regularization_act2)
						elif model_type == "LeNet":
							all_fc1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
							all_fc2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])
							all_fc3_params = torch.cat([x.view(-1) for x in model.fc3.parameters()])
							l2_regularization_fc1 = lambda1 * torch.norm(all_fc1_params, 2)
							l2_regularization_fc2 = lambda2 * torch.norm(all_fc2_params, 2)
							l2_regularization_fc3 = lambda3 * torch.norm(all_fc3_params, 2)
							act_f0, act_f1, act_f2 = activations[-3], activations[-2], activations[-1]
							l2_regularization_actf0 = alambdaf0 * torch.norm(act_f0, 2)
							l2_regularization_actf1 = alambdaf1 * torch.norm(act_f1, 2)
							l2_regularization_actf2 = alambdaf2 * torch.norm(act_f2, 2)
							act_c1, act_c2 = activations[0], activations[1]
							l2_regularization_actc1 = alambdac1 * torch.norm(act_c1, 2)
							l2_regularization_actc2 = alambdac2 * torch.norm(act_c2, 2)
							loss = loss + (l2_regularization_fc1 + l2_regularization_fc2 + l2_regularization_fc3 + l2_regularization_actf0 + l2_regularization_actf1 + l2_regularization_actf2 + l2_regularization_actc1 + l2_regularization_actc2)
					_, preds = torch.max(outputs, 1)
					if phase == 'train':
						loss.backward()
						# during fine-tune step, don't update already pruned gradients
						if finetune:
							if model_type == "MLP1":
								for k, m in enumerate(model.modules()):
									if isinstance(m, nn.Linear):
										weight_copy = m.weight.data.abs().clone()
										mask = weight_copy.gt(0).float()
										m.weight.grad.data.mul_(mask)
							elif model_type == "LeNet":
								# prune fully connected layers
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

			if phase == 'val' and save_dir != None:
				save_model_weights(model, save_dir, "weights_epoch_" + str(epoch) + ".pt")
				if (len(val_accs) == 0 or epoch_acc > max(val_accs)):
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
		save_training_stats(train_losses, train_accs, val_losses, val_accs, save_dir)
		plot_training_stats(train_losses, train_accs, val_losses, val_accs, save_dir)


def prune(model, sparsity_level, save_dir=None, model_type="MLP1"):
	print("Before pruning: ")
	print_parameters_sparsity(model)
	if model_type == "MLP1":
		for m in model.modules():
			if isinstance(m, nn.Linear): # only prune fully connected layers
				if get_sparsity(m.weight.data)[-1] <= sparsity_level:
					continue # already reached desired sparsity
				threshold = get_prune_threshold(m.weight.data, sparsity_level)
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(threshold).float()
				m.weight.data.mul_(mask)
	elif model_type == "LeNet":
		for m in model.modules():
			if isinstance(m, nn.Linear): # prune fully connected layers
				if get_sparsity(m.weight.data)[-1] <= sparsity_level:
					continue # already reached desired sparsity
				threshold = get_prune_threshold(m.weight.data, sparsity_level)
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(threshold).float()
				m.weight.data.mul_(mask)
	print("After pruning: ")
	print_parameters_sparsity(model)
	save_model_weights(model, save_dir)


def finetune(model, num_epochs=10, save_dir=None, l2reg=False, model_type="MLP1"):
	train(model, num_epochs=num_epochs, save_dir=save_dir, l2reg=l2reg, finetune=True, model_type=model_type)

def main():

	parser = argparse.ArgumentParser(description="MLP 1")
	parser.add_argument("--mode", help="mode: train, prune, finetune, eval", type=str, default="eval", nargs="?")
	parser.add_argument("--load_model_name", help="Name of loaded model", type=str, default=None, nargs="?")
	parser.add_argument("--saved_model_name", help="Name of saved model", type=str, default=None, nargs="?")
	parser.add_argument("--num_epochs", help="Number of epochs to train", type=int, default=-1, nargs="?")
	parser.add_argument("--sparsity_level", help="Sparsity level (percentage of non-zero elements", type=float, default=0.1, nargs="?")
	parser.add_argument("--l2reg", help="Add L2 regularization to loss", type=bool, default=False, nargs="?")
	parser.add_argument("--eval_weights", help="Name of weights used in eval", type=str, default="best_weights.pt", nargs="?")
	parser.add_argument("--model_type", help="Model type", type=str, default="MLP1", nargs="?")

	args = parser.parse_args()
	mode, num_epochs, load_model_name, saved_model_name, eval_weights, sparsity_level, l2reg, model_type = args.mode, args.num_epochs, args.load_model_name, args.saved_model_name, args.eval_weights, args.sparsity_level, args.l2reg, args.model_type

	if model_type == "MLP1":
		model = MLP1()
		dir_base_path = "saved_weights/MLP1/"
	elif model_type == "LeNet":
		model = LeNet()
		dir_base_path = "saved_weights/LeNet/"
	else:
		raise("Invalid model type")

	if load_model_name == None:
		load_dir = None
	else:
		load_dir = dir_base_path + str(load_model_name)

	if saved_model_name == None:
		save_dir = None
	else:
		save_dir = dir_base_path + str(saved_model_name)
		try:
			os.stat(save_dir)
		except:
			os.mkdir(save_dir)

	if mode == "eval":
		eval(model, load_dir, eval_weights)
	elif mode == "train":
		if load_dir != None:
			load_model_weights(model, load_dir, eval_weights)
		train(model=model, num_epochs=num_epochs, save_dir=save_dir, l2reg=l2reg, finetune=False, model_type=model_type)
		eval(model, save_dir, eval_weights)
	elif mode == "prune":
		if load_dir != None:
			load_model_weights(model, load_dir, eval_weights)
		prune(model=model, sparsity_level=sparsity_level, savedir=save_dir, model_type=model_type)
		eval(model, save_dir, eval_weights)
	elif mode == "finetune":
		if load_dir != None:
			load_model_weights(model, load_dir, eval_weights)
		finetune(model=model, num_epochs=num_epochs, save_dir=save_dir, l2reg=l2reg, model_type=model_type)
		eval(model, save_dir, eval_weights)
	elif mode == "pf":
		if load_dir != None:
			load_model_weights(model, load_dir, eval_weights)
		print("Before pruning")
		eval(model)
		prune(model=model, sparsity_level=sparsity_level, save_dir=save_dir, model_type=model_type)
		print("After pruning")
		eval(model)
		finetune(model=model, num_epochs=num_epochs, save_dir=save_dir, l2reg=l2reg, model_type=model_type)
		print("After finetuning")
		eval(model)
	else:
		raise("Invalid mode")


if __name__ == '__main__':
	main()