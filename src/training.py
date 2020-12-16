############### Trainin functions operating on implementations of BaseVAE ###############

import os, time, copy
import numpy as np

import torch

# train_vae
# - vae -- instance of class implementing BaseVAE abstract class
# - optimizer -- PyTorch optimizer for all model parameters (will revisit for multi-
#   modal data, which may require separate optimizers for different model parts).
# - data_loaders -- dictionary with keys "train" and "val" pointing to PyTorch 
#   DataLoader objects.
# - num_epochs -- number of training epochs.
# - outfile -- path in which to save state dict of best performing model on val. set.
# Outputs:
# - vae -- model with optimal state dict loaded.
#
def train_vae(vae, optimizer, data_loaders, num_epochs, outfile=None):
	start = time.time()

	best_model_wts = copy.deepcopy(vae.state_dict())
	best_loss = {'loss': np.inf}

	# GPU support
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	vae.to(device)

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
		print('-' * 10, flush=True)

		for phase in ["train", "val"]:
			if phase == "train":
				vae.train()
			else:
				vae.eval()

			epoch_loss = None

			for i,inputs in enumerate(data_loaders[phase]):
				x = inputs[0].to(device)
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase=="train"):
					outputs = vae(x)

					# Weight KLD portion of loss by fraction of dataset being considered.
					M_N = x.size(0) / len(data_loaders[phase].dataset)
					loss = vae.loss_function(*outputs, M_N=M_N)

					if epoch_loss is None:
						epoch_loss = {k:loss[k].item() for k in loss.keys()}
					else:
						for k in loss.keys():
							epoch_loss[k] += loss[k].item()

					if phase == "train":
						loss['loss'].backward()
						optimizer.step()

			for k in epoch_loss.keys():
				epoch_loss[k] = epoch_loss[k] / len(data_loaders[phase])
			loss_str = "\t".join(["%s: %.4f" % (k, epoch_loss[k]) for k in epoch_loss.keys()])
			print("%s %s" % (phase, loss_str), flush=True)

			if phase=="val" and epoch_loss['loss'] < best_loss['loss']:
				best_model_wts = copy.deepcopy(vae.state_dict())
				best_loss = epoch_loss

				if outfile is not None:
					torch.save(vae.state_dict(), outfile)

	elapsed = time.time() - start
	print('Training complete in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60), flush=True)
	loss_str = "\t".join(["%s: %.4f" % (k, best_loss[k]) for k in best_loss.keys()])
	print('Best Model Loss -- %s' % (loss_str), flush=True)

	# load best model weights
	vae.load_state_dict(best_model_wts)
	return vae
