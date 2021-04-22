import torch

from models import MLP1
from main import save_model_weights, load_model_weights
from sparse_util import *

save_dir = 'saved_weights/test1/'
model = MLP1()
load_model_weights(model, save_dir)


# for param in model.parameters():
# 	print(get_sparsity(param))

# mat = model.fc1.weight.data
# mask = get_sparse_mask(mat, 0.01)
# print(mask)
# print(mat * mask)