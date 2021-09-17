# Fine-layered model
# indent = tab
name = "mnist_task"
__python__ = "3.8.5 (GCC 7.3.0:: Anaconda)"
__pytorch__ = "1.7.0"

import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from torchvision import datasets, transforms
import FLmodel_f1_cpp

# Given variable parameters that are refered with args.param_name like args.epochs
parser = argparse.ArgumentParser(description='Pixel-by-pixel MNIST task')
parser.add_argument('--num_hidden_units',type=int,default=128)
parser.add_argument('--num_finelayers',type=int,default=20)
parser.add_argument('--num_layers',type=int,default=10)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--seed',type=int,default=202104160)
parser.add_argument('--outfile',type=str,default='../../Results/Fig9/H128L20_CDcpp.log')
parser.add_argument('--dataset',type=str,default='../../../data')
#
parser.add_argument('--batch_size',type=int,default=100)
parser.add_argument('--lr_rnn',type=float,default=1e-4)
parser.add_argument('--lr_act',type=float,default=1e-5)
parser.add_argument('--lr_out',type=float,default=1e-2)
parser.add_argument('--num_infeatures',type=int,default=1)
parser.add_argument('--num_threads', type=int, default=8)
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

''' (Complex) modReLU activation function '''
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
		self.bias = nn.Parameter(torch.zeros(input_size, dtype=torch.float))
	def forward(self, incmplx):
		return funcCmodReLU.apply(incmplx, self.bias)

''' Complex linear function '''
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

class Diag1(nn.Module):
	def __init__(self, num_features):
		super().__init__()
		self.angle = nn.Parameter(torch.zeros(num_features,1,dtype=torch.float))
	def forward(self, input):
		num_features,batch_size = input.shape
		cosP, sinP = torch.cos(self.angle), torch.sin(self.angle)
		cosP_bcast = cosP.repeat(1,batch_size)
		sinP_bcast = sinP.repeat(1,batch_size)
		outR = cosP_bcast*input.real -sinP_bcast*input.imag
		outI = sinP_bcast*input.real +cosP_bcast*input.imag
		output = torch.complex(outR,outI)
		return output
		
''' Phase_shiter +Directional_coupler: PSDC '''
class funcPSDC(torch.autograd.Function):
	@staticmethod
	def forward(ctx, inX, inY, angle):
		outX, outY = FLmodel_f1_cpp.forwardPSDC(inX,inY,angle)
		ctx.save_for_backward(inX,angle)
		return outX, outY
	@staticmethod
	def backward(ctx, grad_outX, grad_outY):
		inX, angle = ctx.saved_tensors
		grad_inX, grad_inY, grad_angle = \
			FLmodel_f1_cpp.backwardPSDC(grad_outX,grad_outY,inX,angle)
		return grad_inX, grad_inY, grad_angle

class MZIA(nn.Module):
	def __init__(self, num_features):
		super().__init__()
		self.half = int(num_features/2)
		PI = np.pi
		self.angleA0 = nn.Parameter(torch.empty(self.half,dtype=torch.float))
		self.angleA1 = nn.Parameter(torch.empty(self.half,dtype=torch.float))
		nn.init.uniform_(self.angleA0,-PI,PI)
		nn.init.uniform_(self.angleA1,-PI,PI)
	def forward(self, cmplx):
		num_features,batch_size = cmplx.shape
		output = torch.zeros_like(cmplx)
		outX, outY = funcPSDC.apply(cmplx[0:-1:2,:],cmplx[1::2,:],self.angleA0)
		output[0:-1:2,:], output[1::2,:] = funcPSDC.apply(outX,outY,self.angleA1)
		return output

class MZIB(nn.Module):
	def __init__(self, num_features):
		super().__init__()
		self.half = int((num_features-1)/2)
		PI = np.pi
		self.angleB0 = nn.Parameter(torch.empty(self.half,dtype=torch.float))
		self.angleB1 = nn.Parameter(torch.empty(self.half,dtype=torch.float))
		nn.init.uniform_(self.angleB0,-PI,PI)
		nn.init.uniform_(self.angleB1,-PI,PI)
	def forward(self, cmplx):
		num_features,batch_size = cmplx.shape
		output = torch.zeros_like(cmplx)
		output[0,:], output[-1,:] = cmplx[0,:], cmplx[-1,:]
		outX, outY = funcPSDC.apply(cmplx[1:-1:2,:],cmplx[2::2,:],self.angleB0)
		output[1:-1:2,:], output[2::2,:] = funcPSDC.apply(outX,outY,self.angleB1)
		return output

class VariableClementsDiag1(nn.Module):
	def __init__(self, num_features, num_layers):
		super().__init__()
		mlist = []
		for i in range(num_layers):
			if i %2 == 0:
				mlist.append(MZIA(num_features))
			else:
				mlist.append(MZIB(num_features))
		self.module_list = torch.nn.ModuleList(mlist)
		self.diag1 = Diag1(num_features)
	def forward(self, cmplx):
		for f in self.module_list:
			cmplx = f(cmplx)
		output = self.diag1(cmplx)
		return output

class RNNCellCmplx(nn.Module):
	def __init__(self, input_size, hidden_size, init_scale=0.003, bias=True):
		super().__init__()
		self.input_unit = CLinear(input_size,hidden_size,init_scale=0.01,bias=True)
		self.hidden_unit = VariableClementsDiag1(hidden_size,args.num_layers)
		self.add_unit = CmplxAdd()
		self.activation = CmodReLU(hidden_size)
	def forward(self, inputs, state):
		output_ih = self.input_unit(inputs)
		output_hh = self.hidden_unit(state)
		act_in = self.add_unit(output_ih,output_hh)
		state = self.activation(act_in)
		return state

class CmplxState(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.init_stateR = nn.Parameter(torch.zeros(hidden_size,1,dtype=torch.float))
		self.init_stateI = nn.Parameter(torch.zeros(hidden_size,1,dtype=torch.float))
	def forward(self, batch_size):
		init_state_cmplx = torch.complex(self.init_stateR,self.init_stateI)
		state = init_state_cmplx.expand(-1,batch_size)
		return state

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
	with open(args.outfile, 'a') as f:
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
		with open(args.outfile, 'a') as f:
			print('epoch=',epoch+1, file=f)

		loss_list, correct_list = [], []
		for i, (batch_pixels_org,batch_digits) in enumerate(train_loader):
			batch_pixels = batch_pixels_org.view(batch_size, -1)
			batch_pixels_f1 = batch_pixels.permute(1,0).contiguous()
			batch_pixels_f1 = torch.complex(batch_pixels_f1, torch.zeros_like(batch_pixels_f1))
			outputs = network(batch_pixels_f1).permute(1,0)

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
				with open(args.outfile, 'a') as f:
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
	with open(args.outfile, 'a') as f:
		print(f'Test accuracy = {correct/total}', file=f)


if __name__ == "__main__":
	if args.profile:
		with autograd.profiler.profile() as prof:
			main()
		prof.export_chrome_trace("./trace.json")
	else:
		main()
# EOF
