from models import MLP1
from main import save_model_weights, load_model_weights

save_dir = 'saved_weights/test1/'
model = MLP1()
load_model_weights(model, save_dir)
# print(model)

for param in model.parameters():
	print(param.data)