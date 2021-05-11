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

	return ((inputs, activations[0], activations[1], outputs))


def get_MLP1(model_path, save_dir):
	try:
		os.stat(save_dir)
	except:
		os.mkdir(save_dir)
	model = MLP1()
	model.load_state_dict(torch.load(model_path))
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

get_MLP1("saved_weights/MLP1/prune0p01_l2reg/weights_epoch_49.pt", "mtx_weights/MLP1/prune0p01_l2reg_epoch49/")
