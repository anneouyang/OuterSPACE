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

def eval(model, load_dir=None, name="best_weights.pt"):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model.sparsity_level = None

	if load_dir != None:
		load_model_weights(model, load_dir, name)
		print("")
		print("evaluating using model saved at", load_dir+"/"+name, "\n")

	# ignore very small saved weights 
	for m in model.modules():
		if isinstance(m, nn.Linear): # only prune fully connected layers
			weight_copy = m.weight.data.abs().clone()
			mask = weight_copy.gt(1e-2).float()
			m.weight.data.mul_(mask)

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
			outputs, activations = model(inputs, print_activation_sparsity=True)
			loss = criterion(outputs, labels)
			_, preds = torch.max(outputs, 1)

		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(preds == labels.data)

		break

	total_loss = running_loss / 1024
	total_acc = running_corrects.double() / 1024

	print('test loss: {:.4f} acc: {:.4f}'.format(total_loss, total_acc))

	ret = [inputs,]
	for act in activations:
		ret.append(act)
	ret.append(outputs)

	# print([inputs, activations[0], activations[1], outputs])
	# print(ret)
	# raise
	return ret


def get_MLP1(model_path, save_dir):
	try:
		os.stat(save_dir)
	except:
		os.mkdir(save_dir)
	model = MLP1()
	# print_parameters_sparsity(model=model)
	model.load_state_dict(torch.load(model_path))
	# print_parameters_sparsity(model=model)
	acts = eval(model)
	for name, param in model.named_parameters():
		name = name.replace(".", "_")
		# print(name)
		# print(param)
		save_tensor_as_mtx(param.detach(), save_dir+name+".mtx")
	for i in range(len(acts)):
		act = acts[i]
		if i == 0:
			act = act.view(-1, 28 * 28)
		save_tensor_as_mtx(act.detach(), save_dir+"act_"+str(i)+".mtx")
	print_parameters_sparsity(model=model)

def get_LeNet(model_path, save_dir):
	try:
		os.stat(save_dir)
	except:
		os.mkdir(save_dir)
	model = LeNet()
	model.load_state_dict(torch.load(model_path))
	acts = eval(model)
	for name, param in model.named_parameters():
		name = name.replace(".", "_")
		# save conv layers as 
		if "conv" in name and "weight" in name:
			# print(name, param.shape)
			param = param.reshape(param.shape[0], -1)
			# print(name, param.shape)
		save_tensor_as_mtx(param.detach(), save_dir+name+".mtx")
	
	# model1
	# acts: x, xc1, xcp1, xc2, xcp2, xf0, xf1, xf2, xf3
	for i in range(len(acts)):
		act = acts[i]
		if i == 0:
			act = torch.nn.Unfold(kernel_size=(5, 5), dilation=1, padding=2, stride=1)(act)
			act = torch.swapaxes(act, 1, 2)
			act = act.reshape(-1, act.shape[-1])
		elif i == 2:
			act = torch.nn.Unfold(kernel_size=(5, 5), dilation=1, padding=0, stride=1)(act)
			act = torch.swapaxes(act, 1, 2)
			act = act.reshape(-1, act.shape[-1])
		elif i >= 5:
			act = act
		else:
			continue
		print(i, act.shape)
		# x, xcp1, xf0, xf1, xf2, xf3
		save_tensor_as_mtx(act.detach(), save_dir+"act_"+str(i)+".mtx")

	
	




# get_MLP1("saved_weights/MLP1/prune0p01_l2reg/weights_epoch_49.pt", "mtx_weights/MLP1/prune0p01_l2reg_epoch49/")

# get_MLP1("saved_weights/MLP1/no_prune_l2reg/best_weights.pt", "mtx_weights/MLP1/model1/")
# get_MLP1("saved_weights/MLP1/no_prune_no_l2reg/best_weights.pt", "mtx_weights/MLP1/model2/")
# get_MLP1("saved_weights/MLP1/prune0p01_l2reg/weights_epoch_49.pt", "mtx_weights/MLP1/model3/")
# get_MLP1("saved_weights/MLP1/model4/best_weights.pt", "mtx_weights/MLP1/model4-2/")
# get_MLP1("saved_weights/MLP1/model5/best_weights.pt", "mtx_weights/MLP1/model5/")
# get_MLP1("saved_weights/MLP1/model6/best_weights.pt", "mtx_weights/MLP1/model6/")
# get_MLP1("saved_weights/MLP1/model7/best_weights.pt", "mtx_weights/MLP1/model7/")
# get_MLP1("saved_weights/MLP1/model8/best_weights.pt", "mtx_weights/MLP1/model8/")
# get_MLP1("saved_weights/MLP1/model9/best_weights.pt", "mtx_weights/MLP1/model9/")
# get_MLP1("saved_weights/MLP1/model10/best_weights.pt", "mtx_weights/MLP1/model10/")
# get_MLP1("saved_weights/MLP1/scale2/best_weights.pt", "mtx_weights/MLP1/scale2/")
# get_MLP1("saved_weights/MLP1/scale5/best_weights.pt", "mtx_weights/MLP1/scale5/")
# get_MLP1("saved_weights/MLP1/scale10/best_weights.pt", "mtx_weights/MLP1/scale10/")

# get_MLP1("saved_weights/MLP1/prune5/best_weights.pt", "mtx_weights/MLP1/prune5/")
# get_MLP1("saved_weights/MLP1/scale50/best_weights.pt", "mtx_weights/MLP1/scale50/")
# get_MLP1("saved_weights/MLP1/prune50/best_weights.pt", "mtx_weights/MLP1/prune50/")

# get_LeNet("saved_weights/LeNet/no_prune_no_l2reg/weights_epoch_99.pt", "mtx_weights/LeNet/test/")
# get_LeNet("saved_weights/LeNet/model2/best_weights.pt", "mtx_weights/LeNet/model2/")
get_LeNet("saved_weights/LeNet/prune1/best_weights.pt", "mtx_weights/LeNet/prune1/")


