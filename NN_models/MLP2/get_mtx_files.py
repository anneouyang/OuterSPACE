from util import *
from main import eval
from models import *

def get_MLP1(model_path, save_dir):
	try:
		os.stat(save_dir)
	except:
		os.mkdir(save_dir)
	model = MLP1()
	model.load_state_dict(torch.load(model_path))
	eval(model)
	for name, param in model.named_parameters():
		name = name.replace(".", "_")
		# print(name)
		# print(param)
		save_tensor_as_mtx(param.detach(), save_dir+name+".mtx")

get_MLP1("saved_weights/prune0p01_no_l2reg/weights_epoch_49.pt", "mtx_weights/prune0p01_no_l2reg_epoch49/")
