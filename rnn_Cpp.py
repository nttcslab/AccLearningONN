# Fine-layered model: Pytorch C++ Extension
# indent = tab
name = "mnist_task"
__python__ = "3.8.5 (GCC 7.3.0:: Anaconda)"
__pytorch__ = "1.7.0"

import argparse, time, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from torchvision import datasets, transforms
import FLmodel_f1_cpp

default_num_threads = torch.get_num_threads()

# Given variable parameters that are refered with args.param_name like args.epochs
parser = argparse.ArgumentParser(description='Pixel-by-pixel MNIST task')
parser.add_argument('--num_hidden_units',type=int,default=32)
parser.add_argument('--num_finelayers',type=int,default=4)
parser.add_argument('--num_layers',type=int,default=2)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--seed',type=int,default=202104235)
parser.add_argument('--outfile',type=str,default='./Log/rnn_Cpp.log')
parser.add_argument('--dataset',type=str,default='./data')
#
parser.add_argument('--batch_size',type=int,default=100)
parser.add_argument('--lr_rnn',type=float,default=1e-4)
parser.add_argument('--lr_act',type=float,default=1e-5)
parser.add_argument('--lr_out',type=float,default=1e-2)
parser.add_argument('--num_infeatures',type=int,default=1)
parser.add_argument('--num_threads', type=int, default=default_num_threads)
parser.add_argument('--profile',type=str,default=False)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# MNIST
num_classes = 10
train_size = 60000
valid_size = 0
test_size = 10000
num_pixels = 784
maxval_pixel = 255.0

device = 'cpu'
print(f'device = {device}')

''' dataloader '''
class TensorsLoader:
	''' tensors: a list of tensors '''
	def __init__(self, tensors, batch_size=1, shuffle=False):
		self.tensors = tensors
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.data_size = tensors[0].shape[0]
	def __iter__(self):
		self._i = 0
		if self.shuffle:
			index_shuffle = torch.randperm(self.data_size)
			self.tensors = [tensor[index_shuffle] for tensor in self.tensors]
		return self
	def __next__(self):
		i1 = self.batch_size * self._i
		i2 = min( self.batch_size * ( self._i + 1 ), self.data_size )
		if i1 >= self.data_size:
			raise StopIteration()
		self._i += 1
		return [tensor[i1:i2] for tensor in self.tensors]	

''' Complex add function '''
class funcCmplxAdd(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input0, input1):
		output = FLmodel_f1_cpp.forwardCadd(input0,input1)
		return output
	@staticmethod
	def backward(ctx, grad_output):
		grad_input0, grad_input1 = FLmodel_f1_cpp.backwardCadd(grad_output)
		return grad_input0, grad_input1
class CmplxAdd(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, input0, input1):
		return funcCmplxAdd.apply(input0, input1)

''' modReLU activation function '''
class funcCmodReLU(torch.autograd.Function):
	@staticmethod
	def forward(ctx, incmplx, bias):
		ctx.save_for_backward(incmplx,bias)
		output = FLmodel_f1_cpp.forwardmodReLU(incmplx,bias)
		return output
	@staticmethod
	def backward(ctx, grad_output):
		incmplx, bias = ctx.saved_tensors
		grad_input, grad_bias = FLmodel_f1_cpp.backwardmodReLU(grad_output,incmplx,bias)
		return grad_input, grad_bias
class CmodReLU(nn.Module):
	def __init__(self, input_size):
		super().__init__()
		# self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float))
		# self.input_size = input_size
		self.bias = nn.Parameter(torch.zeros(input_size, dtype=torch.float))
	def forward(self, incmplx):
		# bias = self.bias.expand(self.input_size)
		# return funcCmodReLU.apply(incmplx, bias)
		return funcCmodReLU.apply(incmplx, self.bias)

''' (Complex) Diagonal unitary matrix linear function '''
class funcCDiag1(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, omega):
		output = FLmodel_f1_cpp.forwardCDiag1(input,omega)
		ctx.save_for_backward(input,omega)
		return output
	@staticmethod
	def backward(ctx, grad_output):
		input,omega = ctx.saved_tensors
		grad_input,grad_omega = FLmodel_f1_cpp.backwardCDiag1(grad_output,input,omega)
		return grad_input,grad_omega
class CDiag1(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, incmplx, omega):
		return funcCDiag.apply(incmplx,omega)
		
''' C++ Complex linear function '''
class funcCLinear(torch.autograd.Function):
	@staticmethod
	def forward(ctx, incmplx, weight, bias):
		output = FLmodel_f1_cpp.forwardCLinear(incmplx,weight,bias)
		ctx.save_for_backward(incmplx,weight,bias)
		return output
	@staticmethod
	def backward(ctx, grad_output):
		incmplx, weight, bias = ctx.saved_tensors
		grad_incmplx, grad_weight, grad_bias = \
			FLmodel_f1_cpp.backwardCLinear(grad_output,incmplx,weight,bias)
		return grad_incmplx, grad_weight, grad_bias

class CLinear(nn.Module):
	def __init__(self, input_size, output_size, init_scale=1.0, bias=True):
		super().__init__()
		rand_range = init_scale*np.sqrt(1.0/output_size)
		self.weightR = nn.Parameter(init_scale*torch.randn(output_size,input_size,dtype=torch.float))
		self.weightI = nn.Parameter(init_scale*torch.randn(output_size,input_size,dtype=torch.float))
		if bias == True:
			self.biasR = nn.Parameter(init_scale*torch.randn(output_size,dtype=torch.float))
			self.biasI = nn.Parameter(init_scale*torch.randn(output_size,dtype=torch.float))
		else:
			print('ERROR: the setting of bias==False is not allowed.')
			exit()
	def forward(self, cmplx):
		weight = torch.complex(self.weightR, self.weightI)
		bias = torch.complex(self.biasR, self.biasI)
		output = funcCLinear.apply(cmplx,weight,bias)
		return output

''' Clements-structure unitary matrix based on Fang matrices based on PSDCs '''
class funcClementsStrPSDC(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, angleA0, angleA1, angleB0, angleB1):
		outputs = FLmodel_f1_cpp.forwardClementsPSDC(input,angleA0,angleA1,angleB0,angleB1)
		ctx.save_for_backward(input,outputs,angleA0,angleA1,angleB0,angleB1)
		output = outputs[-1]
		return output
	@staticmethod
	def backward(ctx, grad_output):
		input,outputs,angleA0,angleA1,angleB0,angleB1 = ctx.saved_tensors
		grad_angleA0,grad_angleA1,grad_angleB0,grad_angleB1 = \
			FLmodel_f1_cpp.backwardClementsPSDC(grad_output,outputs,input,angleA0,angleA1,angleB0,angleB1)
		grad_input = grad_output
		return grad_input,grad_angleA0,grad_angleA1,grad_angleB0,grad_angleB1	

''' Number_of_Layers-variable Clements-structure unitary matrix +CDiag1 '''
class VariableClementsCDiag1(nn.Module):
	def __init__(self, num_features, num_layers):
		super().__init__()
		PI = np.pi
		self.nFeatures = num_features
		self.nAnglesA = int(num_features/2)
		self.nAnglesB = int((num_features-1)/2)
		self.nLayersB = int(num_layers/2)
		self.nLayersA = num_layers -self.nLayersB
		self.angleA0 = nn.Parameter(torch.empty(self.nLayersA,self.nAnglesA))
		self.angleA1 = nn.Parameter(torch.empty(self.nLayersA,self.nAnglesA))
		self.angleB0 = nn.Parameter(torch.empty(self.nLayersB,self.nAnglesB))
		self.angleB1 = nn.Parameter(torch.empty(self.nLayersB,self.nAnglesB))
		nn.init.uniform_(self.angleA0,-PI,PI)
		nn.init.uniform_(self.angleA1,-PI,PI)
		nn.init.uniform_(self.angleB0,-PI,PI)
		nn.init.uniform_(self.angleB1,-PI,PI)
		self.omega = nn.Parameter(torch.zeros(self.nFeatures,dtype=torch.float))
	def forward(self, cmplx):
		interm = funcClementsStrPSDC.apply(cmplx,self.angleA0,self.angleA1,self.angleB0,self.angleB1)
		output = funcCDiag1.apply(interm,self.omega)
		return output

''' Complex RNN cell '''
class RNNCellCmplx(nn.Module):
	def __init__(self, input_size, hidden_size, init_scale=0.003, bias=True):
		super().__init__()
		self.input_unit = CLinear(input_size,hidden_size,init_scale=0.01,bias=True)
		self.hidden_unit = VariableClementsCDiag1(hidden_size,args.num_layers)
		self.add_unit = CmplxAdd()
		self.activation = CmodReLU(hidden_size)
	def forward(self, inputs, state):
		output_ih = self.input_unit(inputs)
		output_hh = self.hidden_unit(state)
		act_in = self.add_unit(output_ih,output_hh)
		state = self.activation(act_in)
		#state = self.activation(output_ih+output_hh)
		return state

''' Complex initial state for hidden unit '''
class CmplxState(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.init_stateR = nn.Parameter(torch.zeros(hidden_size,1,dtype=torch.float))
		# self.init_stateR = torch.ones(hidden_size,1,dtype=torch.float)/np.sqrt(hidden_size)
		self.init_stateI = nn.Parameter(torch.zeros(hidden_size,1,dtype=torch.float))
	def forward(self, batch_size):
		init_state_cmplx = torch.complex(self.init_stateR,self.init_stateI)
		state = init_state_cmplx.expand(-1,batch_size)
		return state

''' Modified Elman-type RNN '''
class modElmanNN(nn.Module):
	def __init__(self, num_infeatures, hidden_size, num_classes):
		super().__init__()
		self.rnn = RNNCellCmplx(num_infeatures,hidden_size,bias=True)
		self.init_state = CmplxState(hidden_size)
		self.outLinearNN = CLinear(hidden_size,num_classes,init_scale=0.1,bias=True)
		self.loss_func = nn.CrossEntropyLoss()
	def forward(self, inputs):
		num_pixels,batch_size = inputs.shape
		inputs_ext = inputs[:,None,:]
		state = self.init_state(batch_size)
		for idx in range(num_pixels):
			inputs_pixel = inputs_ext[idx]
			state = self.rnn(inputs_pixel,state)
		output = self.outLinearNN(state)
		out_power = output.abs()**2	
		return out_power

	def criterion(self, outputs, digits):
		return self.loss_func(outputs,digits)

	def correct_count(self, outputs, digits):
		return torch.eq(torch.argmax(outputs,dim=1),digits).float().sum()

def main():
	torch.set_num_threads(args.num_threads)
	print(torch.__config__.parallel_info())	
	print(args)
	with open(args.outfile,'a') as f:
		print(torch.__config__.parallel_info(), file=f)	
		print(args, file=f)

	# Initialization
	num_infeatures = args.num_infeatures
	num_hidden_units = args.num_hidden_units
	network = modElmanNN(num_infeatures,num_hidden_units,num_classes)
	optimizer = optim.RMSprop([{'params':network.rnn.input_unit.parameters(), 'lr':args.lr_rnn},
						{'params':network.rnn.hidden_unit.parameters(), 'lr':args.lr_rnn},
						{'params':network.rnn.activation.parameters(), 'lr':args.lr_act},
						{'params':network.outLinearNN.parameters(), 'lr':args.lr_out}], eps=1e-5)

	# Training
	batch_size = args.batch_size
	max_size = train_size/batch_size
	mnist = datasets.MNIST(root=args.dataset,train=True,download=True,transform=transforms.ToTensor())
	mnist_images = mnist.data/maxval_pixel
	mnist_targets = mnist.targets
	train_loader = TensorsLoader([mnist_images,mnist_targets],batch_size=batch_size,shuffle=True)

	num_epochs = args.epochs
	start_time = time.time()
	for epoch in range(num_epochs):
		print('epoch=',epoch+1)
		with open(args.outfile,'a') as f:
			print('epoch=',epoch+1, file=f)

		loss_list, correct_list = [], []
		for i, (batch_pixels_org,batch_digits) in enumerate(train_loader):
			batch_pixels = batch_pixels_org.view(batch_size, -1)
			batch_pixels_f1 = batch_pixels.permute(1,0).contiguous() # feature first
			batch_pixels_f1 = torch.complex(batch_pixels_f1, torch.zeros_like(batch_pixels_f1))
			outputs = network(batch_pixels_f1).permute(1,0) # batch first

			loss = network.criterion(outputs,batch_digits)
			correct = network.correct_count(outputs,batch_digits)
			loss_list.append(loss.item()), correct_list.append(correct.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			progress = 10*(i+1)/max_size
			if progress.is_integer() == True:
				loss_mean, correct_mean = np.mean(loss_list), np.mean(correct_list)
				elapsed_time = time.time() - start_time
				print('elapsed_time: {:.2f}'.format(elapsed_time), \
					' processed: {:.1f}%'.format(100*((i+1)*batch_size)/train_size),\
					' loss={:.6f}'.format(loss_mean), \
					' acc={:.5f}'.format(correct_mean/batch_size))
				with open(args.outfile,'a') as f:
					print('elapsed_time: {:.2f}'.format(elapsed_time), \
						' processed: {:.1f}%'.format(100*((i+1)*batch_size)/train_size),\
						' loss={:.6f}'.format(loss_mean), \
						' acc={:.5f}'.format(correct_mean/batch_size), file=f)

	# Test
	mnist_test = datasets.MNIST(root=args.dataset,train=False,download=True,transform=transforms.ToTensor())
	test_images = mnist_test.data.view(-1,num_pixels)/maxval_pixel
	test_loader = TensorsLoader([test_images,mnist_test.targets],batch_size=batch_size,shuffle=True)
	correct = 0
	total = 0
	for i, (batch_pixels,batch_digits) in enumerate(test_loader,0):
		batch_pixels_f1 = batch_pixels.permute(1,0).contiguous() # feature first
		batch_pixels_f1 = torch.complex(batch_pixels_f1,torch.zeros_like(batch_pixels_f1))
		output = network(batch_pixels_f1).permute(1,0) # batch first
		out_power = output.abs()**2
		_, predicted = torch.max(out_power.data, dim=1)
		total += batch_digits.size(0)
		correct += (predicted == batch_digits).sum().item()
	print(f'Test accuracy = {correct/total}')
	with open(args.outfile,'a') as f:
		print(f'Test accuracy = {correct/total}', file=f)


if __name__ == "__main__":
	if args.profile:
		with autograd.profiler.profile() as prof:
			main()
		prof.export_chrome_trace("./trace.json")
	else:
		main()
## EOF
