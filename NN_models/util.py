import torch
import torch.nn as nn

import copy
import os
import sys
import copy

import pickle
import matplotlib.pyplot as plt
import numpy as np

import scipy.io


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


def save_tensor_as_mtx(a, save_file):
	scipy.io.mmwrite(save_file, scipy.sparse.csr_matrix(a.numpy()))
