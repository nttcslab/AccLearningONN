#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>

void set_num_threads(int num_threads){
	omp_set_num_threads(num_threads);
	at::set_num_threads(num_threads);
	std::cout << "omp_num_threads = " << omp_get_num_threads() << std::endl;
	std::cout << "at::num_threads = " << at::get_num_threads() << std::endl;
}

template<typename T>
c10::complex<T> conj(c10::complex<T> z)
{return c10::complex<T>(z.real(),-z.imag());}

template<typename T>
c10::complex<T> cmplxadd(c10::complex<T> a, c10::complex<T> b)
{return c10::complex<T>(a.real()+b.real(),a.imag()+b.imag());}



// Clements-structure (PSDC)^2
void forwardPSDC_layer(
	torch::TensorAccessor<c10::complex<float>,2> output_a, 
	torch::TensorAccessor<c10::complex<float>,2> input_a, 
	torch::TensorAccessor<float,1> angle_a, int offset)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const int nFeatures_m1 = input_a.size(0)-1;
	const int nSamples = input_a.size(1);
	const int nAngles = angle_a.size(0);

	float invSqrt2 = 1.0/sqrt(2.0);
	at::parallel_for(0, nSamples, 0, [&](int64_t start, int64_t end)
	{
		for(int64_t s = start; s < end; ++s){
			output_a[0][s] = input_a[0][s];
			output_a[nFeatures_m1][s] = input_a[nFeatures_m1][s];
		}
	});
	at::parallel_for(0, nAngles, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			int idx0 = offset +2*h, idx1 = idx0+1;
			c10::complex<float> exp_iangle = exp(1._if*angle_a[h]);
			for(int s = 0; s < nSamples; ++s){
				c10::complex<float> val = input_a[idx0][s]*exp_iangle;
				output_a[idx0][s] = invSqrt2*(val +1._if*input_a[idx1][s]);
				output_a[idx1][s] = invSqrt2*(1._if*val +input_a[idx1][s]);
			}
		}
	});
}
void backwardPSDC_layer(
	torch::TensorAccessor<c10::complex<float>,2> grad_output_a,
	torch::TensorAccessor<float,1> grad_angle_a,
	torch::TensorAccessor<c10::complex<float>,2> input_a,
	torch::TensorAccessor<float,1> angle_a, int offset)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const int nSamples = input_a.size(1);
	const int nAngles = angle_a.size(0);
	
	float invSqrt2 = (float) 1.0f/sqrt(2.0);
	at::parallel_for(0, nAngles, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			int idx0 = offset +2*h, idx1 = idx0+1;
			auto gout0 = grad_output_a[idx0];
			auto gout1 = grad_output_a[idx1];
			c10::complex<float> val_exp = exp(1._if*(-angle_a[h]));
			c10::complex<float> gout0_s = 0, gout1_s = 0;
			float grad_angle_h = 0;
			for(int s = 0; s < nSamples; ++s){
				gout0_s = invSqrt2*val_exp*(gout0[s] -1._if*gout1[s]);
				gout1_s = invSqrt2*(-1._if*gout0[s] +gout1[s]);
				grad_angle_h += 2.0f*(conj(input_a[idx0][s])*gout0_s).imag();
				gout0[s] = gout0_s;
				gout1[s] = gout1_s;
			}
			grad_angle_a[h] = grad_angle_h;
		}
	});
}
torch::Tensor forwardClementsPSDC(
	torch::Tensor input, 
	torch::Tensor angleA0, torch::Tensor angleA1, 
	torch::Tensor angleB0, torch::Tensor angleB1)
{
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angleA0_a = angleA0.accessor<float,2>();
	const auto angleA1_a = angleA1.accessor<float,2>();
	const auto angleB0_a = angleB0.accessor<float,2>();
	const auto angleB1_a = angleB1.accessor<float,2>();
	const int nLayersA = angleA0_a.size(0);
	const int nLayersB = angleB0_a.size(0);
	const int nFineLayers = 2*(nLayersA +nLayersB);
	const int nFeatures = input_a.size(0); 
	const int nSamples = input_a.size(1); // Batch size
	auto outputs = torch::empty({nFineLayers,nFeatures,nSamples},c10::kComplexFloat);
	auto outputs_a = outputs.accessor<c10::complex<float>,3>();
	auto input_layer = input_a;

	for(int finelayer = 0; finelayer < nFineLayers; ++finelayer){
		auto output_layer = outputs_a[finelayer];
		int layer = (int) std::floor(finelayer/2);
		int ABidx = (int) std::floor(layer/2); // Index of (A,B) tuples
		int mod_finelayer = finelayer %4;
		switch(mod_finelayer){
			case 0:
				forwardPSDC_layer(output_layer,input_layer,angleA0_a[ABidx],0);
				break;
			case 1:
				forwardPSDC_layer(output_layer,input_layer,angleA1_a[ABidx],0);
				break;
			case 2:
				forwardPSDC_layer(output_layer,input_layer,angleB0_a[ABidx],1);
				break;
			case 3:
				forwardPSDC_layer(output_layer,input_layer,angleB1_a[ABidx],1);
				break;
			default:
				printf("ERROR: Unknown finelayer ID, mod=%d\n",mod_finelayer);
				exit(1);
		}
		input_layer = output_layer;
	}
	return outputs;
}
std::vector<at::Tensor> backwardClementsPSDC(
	torch::Tensor grad_output, torch::Tensor outputs, torch::Tensor input,
	torch::Tensor angleA0, torch::Tensor angleA1, 
	torch::Tensor angleB0, torch::Tensor angleB1)
{
	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const auto outputs_a = outputs.accessor<c10::complex<float>,3>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angleA0_a = angleA0.accessor<float,2>();
	const auto angleA1_a = angleA1.accessor<float,2>();
	const auto angleB0_a = angleB0.accessor<float,2>();
	const auto angleB1_a = angleB1.accessor<float,2>();
	const int nLayersA = angleA0_a.size(0);
	const int nLayersB = angleB0_a.size(0);
	const int nFineLayers = 2*(nLayersA +nLayersB);
	const int nAnglesA = angleA0_a.size(1);
	const int nAnglesB = angleB0_a.size(1);
	auto grad_angleA0 = torch::zeros({nLayersA,nAnglesA}, at::kFloat);
	auto grad_angleA0_a = grad_angleA0.accessor<float,2>();
	auto grad_angleA1 = torch::zeros({nLayersA,nAnglesA}, at::kFloat);
	auto grad_angleA1_a = grad_angleA1.accessor<float,2>();
	auto grad_angleB0 = torch::zeros({nLayersB,nAnglesB}, at::kFloat);
	auto grad_angleB0_a = grad_angleB0.accessor<float,2>();
	auto grad_angleB1 = torch::zeros({nLayersB,nAnglesB}, at::kFloat);
	auto grad_angleB1_a = grad_angleB1.accessor<float,2>();

	for(int finelayer = nFineLayers-1; finelayer >= 0; --finelayer){
		auto input_layer = (finelayer == 0) ? input_a : outputs_a[finelayer -1];
		int layer = (int) std::floor(finelayer/2);
		int ABidx = (int) std::floor(layer/2); //Index of (A,B) tuples
		int mod_finelayer = finelayer %4;
		switch(mod_finelayer){
			case 3:
				backwardPSDC_layer(grad_output_a,grad_angleB1_a[ABidx],input_layer,angleB1_a[ABidx],1);
				break;
			case 2:
				backwardPSDC_layer(grad_output_a,grad_angleB0_a[ABidx],input_layer,angleB0_a[ABidx],1);
				break;
			case 1:
				backwardPSDC_layer(grad_output_a,grad_angleA1_a[ABidx],input_layer,angleA1_a[ABidx],0);
				break;
			case 0:
				backwardPSDC_layer(grad_output_a,grad_angleA0_a[ABidx],input_layer,angleA0_a[ABidx],0);
				break;
			default:
				printf("ERROR: Unknown finelayer ID, mod=%d\n",mod_finelayer);
				exit(1);
		}
	}
	return {grad_angleA0, grad_angleA1, grad_angleB0, grad_angleB1};
}


// Clements-structure (DCPS)^2
void forwardDCPS_layer(
	torch::TensorAccessor<c10::complex<float>,2> output_a, 
	torch::TensorAccessor<c10::complex<float>,2> input_a, 
	torch::TensorAccessor<float,1> angle_a, int offset)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const int nFeatures_m1 = input_a.size(0)-1;
	const int nSamples = input_a.size(1);
	const int nAngles = angle_a.size(0);

	float invSqrt2 = 1.0/sqrt(2.0);
	at::parallel_for(0, nSamples, 0, [&](int64_t start, int64_t end)
	{
		for(int64_t s = start; s < end; ++s){
			output_a[0][s] = input_a[0][s];
			output_a[nFeatures_m1][s] = input_a[nFeatures_m1][s];
		}
	});
	at::parallel_for(0, nAngles, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			int idx0 = offset +2*h, idx1 = idx0+1;
			c10::complex<float> exp_iangle = exp(1._if*angle_a[h]);
			for(int s = 0; s < nSamples; ++s){
				output_a[idx0][s] = invSqrt2*exp_iangle*(input_a[idx0][s] +1._if*input_a[idx1][s]);
				output_a[idx1][s] = invSqrt2*(1._if*input_a[idx0][s] +input_a[idx1][s]);
			}
		}
	});
}
void backwardDCPS_layer(
	torch::TensorAccessor<c10::complex<float>,2> grad_output_a,
	torch::TensorAccessor<float,1> grad_angle_a,
	torch::TensorAccessor<c10::complex<float>,2> output_a,
	torch::TensorAccessor<float,1> angle_a, int offset)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const int nSamples = output_a.size(1);
	const int nAngles = angle_a.size(0);
	
	float invSqrt2 = 1.0f/sqrt(2.0);
	at::parallel_for(0, nAngles, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			int idx0 = offset +2*h, idx1 = idx0+1;
			auto gout0 = grad_output_a[idx0];
			auto gout1 = grad_output_a[idx1];
			c10::complex<float> exp_miangle = exp(1._if*(-angle_a[h]));
			c10::complex<float> gout0_s = 0, gout1_s = 0;
			float grad_angle_h = 0;
			for(int s = 0; s < nSamples; ++s){
				grad_angle_h += 2.0f*(conj(output_a[idx0][s])*gout0[s]).imag();
				gout0_s = invSqrt2*(exp_miangle*gout0[s] -1._if*gout1[s]);
				gout1_s = invSqrt2*(-1._if*exp_miangle*gout0[s] +gout1[s]);
				gout0[s] = gout0_s;
				gout1[s] = gout1_s;
			}
			grad_angle_a[h] = grad_angle_h;
		}
	});
}
torch::Tensor forwardClementsDCPS(
	torch::Tensor input, 
	torch::Tensor angleA0, torch::Tensor angleA1, 
	torch::Tensor angleB0, torch::Tensor angleB1)
{
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angleA0_a = angleA0.accessor<float,2>();
	const auto angleA1_a = angleA1.accessor<float,2>();
	const auto angleB0_a = angleB0.accessor<float,2>();
	const auto angleB1_a = angleB1.accessor<float,2>();
	const int nLayersA = angleA0_a.size(0);
	const int nLayersB = angleB0_a.size(0);
	const int nFineLayers = 2*(nLayersA +nLayersB);
	const int nFeatures = input_a.size(0); 
	const int nSamples = input_a.size(1); // Batch size
	auto outputs = torch::empty({nFineLayers,nFeatures,nSamples},c10::kComplexFloat);
	auto outputs_a = outputs.accessor<c10::complex<float>,3>();
	auto input_layer = input_a;

	for(int finelayer = 0; finelayer < nFineLayers; ++finelayer){
		auto output_layer = outputs_a[finelayer];
		int layer = (int) std::floor(finelayer/2);
		int ABidx = (int) std::floor(layer/2); // Index of (A,B) tuples
		int mod_finelayer = finelayer %4;
		switch(mod_finelayer){
			case 0:
				forwardDCPS_layer(output_layer,input_layer,angleA0_a[ABidx],0);
				break;
			case 1:
				forwardDCPS_layer(output_layer,input_layer,angleA1_a[ABidx],0);
				break;
			case 2:
				forwardDCPS_layer(output_layer,input_layer,angleB0_a[ABidx],1);
				break;
			case 3:
				forwardDCPS_layer(output_layer,input_layer,angleB1_a[ABidx],1);
				break;
			default:
				printf("ERROR: Unknown finelayer ID, mod=%d\n",mod_finelayer);
				exit(1);
		}
		input_layer = output_layer;
	}
	return outputs;
}
std::vector<at::Tensor> backwardClementsDCPS(
	torch::Tensor grad_output, torch::Tensor outputs, torch::Tensor input,
	torch::Tensor angleA0, torch::Tensor angleA1, 
	torch::Tensor angleB0, torch::Tensor angleB1)
{
	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const auto outputs_a = outputs.accessor<c10::complex<float>,3>();
	const auto angleA0_a = angleA0.accessor<float,2>();
	const auto angleA1_a = angleA1.accessor<float,2>();
	const auto angleB0_a = angleB0.accessor<float,2>();
	const auto angleB1_a = angleB1.accessor<float,2>();
	const int nLayersA = angleA0_a.size(0);
	const int nLayersB = angleB0_a.size(0);
	const int nFineLayers = 2*(nLayersA +nLayersB);
	const int nAnglesA = angleA0_a.size(1);
	const int nAnglesB = angleB0_a.size(1);
	auto grad_angleA0 = torch::zeros({nLayersA,nAnglesA}, at::kFloat);
	auto grad_angleA0_a = grad_angleA0.accessor<float,2>();
	auto grad_angleA1 = torch::zeros({nLayersA,nAnglesA}, at::kFloat);
	auto grad_angleA1_a = grad_angleA1.accessor<float,2>();
	auto grad_angleB0 = torch::zeros({nLayersB,nAnglesB}, at::kFloat);
	auto grad_angleB0_a = grad_angleB0.accessor<float,2>();
	auto grad_angleB1 = torch::zeros({nLayersB,nAnglesB}, at::kFloat);
	auto grad_angleB1_a = grad_angleB1.accessor<float,2>();

	for(int finelayer = nFineLayers-1; finelayer >= 0; --finelayer){
		auto output_layer = outputs_a[finelayer];
		int layer = (int) std::floor(finelayer/2);
		int ABidx = (int) std::floor(layer/2); //Index of (A,B) tuples
		int mod_finelayer = finelayer %4;
		switch(mod_finelayer){
			case 3:
				backwardDCPS_layer(grad_output_a,grad_angleB1_a[ABidx],output_layer,angleB1_a[ABidx],1);
				break;
			case 2:
				backwardDCPS_layer(grad_output_a,grad_angleB0_a[ABidx],output_layer,angleB0_a[ABidx],1);
				break;
			case 1:
				backwardDCPS_layer(grad_output_a,grad_angleA1_a[ABidx],output_layer,angleA1_a[ABidx],0);
				break;
			case 0:
				backwardDCPS_layer(grad_output_a,grad_angleA0_a[ABidx],output_layer,angleA0_a[ABidx],0);
				break;
			default:
				printf("ERROR: Unknown finelayer ID, mod=%d\n",mod_finelayer);
				exit(1);
		}
	}
	return {grad_angleA0, grad_angleA1, grad_angleB0, grad_angleB1};
}


// Clements-structure (DCPS)_(PSDC) == DBLarm
void forwardLPSDC_layer(// PS at lower arm
	torch::TensorAccessor<c10::complex<float>,2> output_a, 
	torch::TensorAccessor<c10::complex<float>,2> input_a, 
	torch::TensorAccessor<float,1> angle_a, int offset)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const int nFeatures_m1 = input_a.size(0)-1;
	const int nSamples = input_a.size(1);
	const int nAngles = angle_a.size(0);

	float invSqrt2 = 1.0/sqrt(2.0);
	at::parallel_for(0, nSamples, 0, [&](int64_t start, int64_t end)
	{
		for(int64_t s = start; s < end; ++s){
			output_a[0][s] = input_a[0][s];
			output_a[nFeatures_m1][s] = input_a[nFeatures_m1][s];
		}
	});
	at::parallel_for(0, nAngles, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			int idx0 = offset +2*h, idx1 = idx0+1;
			c10::complex<float> exp_iangle = exp(1._if*angle_a[h]);
			for(int s = 0; s < nSamples; ++s){
				c10::complex<float> val = exp_iangle*input_a[idx1][s];
				output_a[idx0][s] = invSqrt2*(input_a[idx0][s] +1._if*val);
				output_a[idx1][s] = invSqrt2*(1._if*input_a[idx0][s] +val);
			}
		}
	});
}
void backwardLPSDC_layer(// PS at lower arm
	torch::TensorAccessor<c10::complex<float>,2> grad_output_a,
	torch::TensorAccessor<float,1> grad_angle_a,
	torch::TensorAccessor<c10::complex<float>,2> input_a,
	torch::TensorAccessor<float,1> angle_a, int offset)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const int nSamples = input_a.size(1);
	const int nAngles = angle_a.size(0);
	
	float invSqrt2 = (float) 1.0f/sqrt(2.0);
	at::parallel_for(0, nAngles, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			int idx0 = offset +2*h, idx1 = idx0+1;
			auto gout0 = grad_output_a[idx0];
			auto gout1 = grad_output_a[idx1];
			c10::complex<float> exp_miangle = exp(1._if*(-angle_a[h]));
			c10::complex<float> gout0_s = 0, gout1_s = 0;
			float grad_angle_h = 0;
			for(int s = 0; s < nSamples; ++s){
				gout0_s = invSqrt2*(gout0[s] -1._if*gout1[s]);
				gout1_s = invSqrt2*exp_miangle*(-1._if*gout0[s] +gout1[s]);
				grad_angle_h += 2.0f*(conj(input_a[idx1][s])*gout1_s).imag();
				gout0[s] = gout0_s;
				gout1[s] = gout1_s;
			}
			grad_angle_a[h] = grad_angle_h;
		}
	});
}
torch::Tensor forwardClementsDBLarm(
	torch::Tensor input, 
	torch::Tensor angleA0, torch::Tensor angleA1, 
	torch::Tensor angleB0, torch::Tensor angleB1)
{
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angleA0_a = angleA0.accessor<float,2>();
	const auto angleA1_a = angleA1.accessor<float,2>();
	const auto angleB0_a = angleB0.accessor<float,2>();
	const auto angleB1_a = angleB1.accessor<float,2>();
	const int nLayersA = angleA0_a.size(0);
	const int nLayersB = angleB0_a.size(0);
	const int nFineLayers = 2*(nLayersA +nLayersB);
	const int nFeatures = input_a.size(0); 
	const int nSamples = input_a.size(1); // Batch size
	auto outputs = torch::empty({nFineLayers,nFeatures,nSamples},c10::kComplexFloat);
	auto outputs_a = outputs.accessor<c10::complex<float>,3>();
	auto input_layer = input_a;

	for(int finelayer = 0; finelayer < nFineLayers; ++finelayer){
		auto output_layer = outputs_a[finelayer];
		int layer = (int) std::floor(finelayer/2);
		int ABidx = (int) std::floor(layer/2); // Index of (A,B) tuples
		int mod_finelayer = finelayer %4;
		switch(mod_finelayer){
			case 0:
				forwardDCPS_layer(output_layer,input_layer,angleA0_a[ABidx],0);
				break;
			case 1:
				forwardLPSDC_layer(output_layer,input_layer,angleA1_a[ABidx],0);
				break;
			case 2:
				forwardDCPS_layer(output_layer,input_layer,angleB0_a[ABidx],1);
				break;
			case 3:
				forwardLPSDC_layer(output_layer,input_layer,angleB1_a[ABidx],1);
				break;
			default:
				printf("ERROR: Unknown finelayer ID, mod=%d\n",mod_finelayer);
				exit(1);
		}
		input_layer = output_layer;
	}
	return outputs;
}
std::vector<at::Tensor> backwardClementsDBLarm(
	torch::Tensor grad_output, torch::Tensor outputs, torch::Tensor input,
	torch::Tensor angleA0, torch::Tensor angleA1, 
	torch::Tensor angleB0, torch::Tensor angleB1)
{
	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const auto outputs_a = outputs.accessor<c10::complex<float>,3>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angleA0_a = angleA0.accessor<float,2>();
	const auto angleA1_a = angleA1.accessor<float,2>();
	const auto angleB0_a = angleB0.accessor<float,2>();
	const auto angleB1_a = angleB1.accessor<float,2>();
	const int nLayersA = angleA0_a.size(0);
	const int nLayersB = angleB0_a.size(0);
	const int nFineLayers = 2*(nLayersA +nLayersB);
	const int nAnglesA = angleA0_a.size(1);
	const int nAnglesB = angleB0_a.size(1);
	auto grad_angleA0 = torch::zeros({nLayersA,nAnglesA}, at::kFloat);
	auto grad_angleA0_a = grad_angleA0.accessor<float,2>();
	auto grad_angleA1 = torch::zeros({nLayersA,nAnglesA}, at::kFloat);
	auto grad_angleA1_a = grad_angleA1.accessor<float,2>();
	auto grad_angleB0 = torch::zeros({nLayersB,nAnglesB}, at::kFloat);
	auto grad_angleB0_a = grad_angleB0.accessor<float,2>();
	auto grad_angleB1 = torch::zeros({nLayersB,nAnglesB}, at::kFloat);
	auto grad_angleB1_a = grad_angleB1.accessor<float,2>();

	for(int finelayer = nFineLayers-1; finelayer >= 0; --finelayer){
		auto input_layer = (finelayer == 0) ? input_a : outputs_a[finelayer -1];
		int layer = (int) std::floor(finelayer/2);
		int ABidx = (int) std::floor(layer/2); //Index of (A,B) tuples
		int mod_finelayer = finelayer %4;
		switch(mod_finelayer){
			case 3:
				backwardLPSDC_layer(grad_output_a,grad_angleB1_a[ABidx],input_layer,angleB1_a[ABidx],1);
				break;
			case 2:
				backwardDCPS_layer(grad_output_a,grad_angleB0_a[ABidx],input_layer,angleB0_a[ABidx],1);
				break;
			case 1:
				backwardLPSDC_layer(grad_output_a,grad_angleA1_a[ABidx],input_layer,angleA1_a[ABidx],0);
				break;
			case 0:
				backwardDCPS_layer(grad_output_a,grad_angleA0_a[ABidx],input_layer,angleA0_a[ABidx],0);
				break;
			default:
				printf("ERROR: Unknown finelayer ID, mod=%d\n",mod_finelayer);
				exit(1);
		}
	}
	return {grad_angleA0, grad_angleA1, grad_angleB0, grad_angleB1};
}


// PSDC for CDcpp
std::vector<at::Tensor> forwardPSDC(
	torch::Tensor inX, torch::Tensor inY, torch::Tensor angle)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto inX_a = inX.accessor<c10::complex<float>,2>();
	const auto inY_a = inY.accessor<c10::complex<float>,2>();
	const auto angle_a = angle.accessor<float,1>();
	const int nFeatures = inX_a.size(0);
	const int nSamples = inX_a.size(1);
	auto outX = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto outX_a = outX.accessor<c10::complex<float>,2>();
	auto outY = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto outY_a = outY.accessor<c10::complex<float>,2>();
	auto exp_iangle = torch::empty({nFeatures},c10::kComplexFloat);
	auto exp_iangle_a = exp_iangle.accessor<c10::complex<float>,1>();

	float invSqrt2 = (float) 1.0f/sqrt(2.0);
	at::parallel_for(0, nFeatures, 0, [&](int64_t start, int64_t end){
		for(int64_t i = start; i < end; ++i){
			auto inX_i = inX_a[i], inY_i = inY_a[i];
			auto outX_i = outX_a[i], outY_i = outY_a[i];
			exp_iangle_a[i] = exp(1._if*angle_a[i]);
			c10::complex<float> val = 0;
			for(int s = 0; s < nSamples; ++s){
				val = inX_i[s]*exp_iangle_a[i];
				outX_i[s] = invSqrt2*(val +1._if*inY_i[s]);
				outY_i[s] = invSqrt2*(1._if*val +inY_i[s]);
			}
		}
	});
	return {outX, outY};
}
std::vector<at::Tensor> backwardPSDC(
	torch::Tensor grad_outX, torch::Tensor grad_outY,
	torch::Tensor inX, torch::Tensor angle)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto grad_outX_a = grad_outX.accessor<c10::complex<float>,2>();
	const auto grad_outY_a = grad_outY.accessor<c10::complex<float>,2>();
	const auto inX_a = inX.accessor<c10::complex<float>,2>();
	const auto angle_a = angle.accessor<float,1>();
	const int nFeatures = inX_a.size(0);
	const int nSamples = inX_a.size(1);
	auto grad_inX = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto grad_inX_a = grad_inX.accessor<c10::complex<float>,2>();
	auto grad_inY = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto grad_inY_a = grad_inY.accessor<c10::complex<float>,2>();
	auto grad_angle = torch::zeros({nFeatures},at::kFloat);
	auto grad_angle_a = grad_angle.accessor<float,1>();

	float invSqrt2 = (float) 1.0f/sqrt(2.0);
	at::parallel_for(0, nFeatures, 0, [&](int64_t start, int64_t end){
		for(int64_t i = start; i < end; ++i){
			auto grad_outX_i = grad_outX_a[i], grad_outY_i = grad_outY_a[i];
			auto grad_inX_i = grad_inX_a[i], grad_inY_i = grad_inY_a[i];
			auto inX_i = inX_a[i];
			c10::complex<float> val_exp = exp(-1._if*angle_a[i]);
			for(int s = 0; s < nSamples; ++s){
				grad_inX_i[s] = invSqrt2*val_exp*(grad_outX_i[s] -1._if*grad_outY_i[s]);
				grad_inY_i[s] = invSqrt2*(-1._if*grad_outX_i[s] +grad_outY_i[s]);
				grad_angle_a[i] += 2.0f*(conj(inX_i[s])*grad_inX_i[s]).imag();
			}
		}
	});
	return {grad_inX, grad_inY, grad_angle};
}


// (Complex) modReLU
torch::Tensor forwardmodReLU(
	torch::Tensor input, torch::Tensor bias)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto bias_a = bias.accessor<float,1>();
	const int nFeatures = input_a.size(0);
	const int nSamples = input_a.size(1);
	auto output = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto output_a = output.accessor<c10::complex<float>,2>();
	const float eps = 1e-5; // epsilon: Magic number

	at::parallel_for(0, nFeatures, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			c10::complex<float> incmplx = 0;
			float norm = 0, scale = 0;
			float bias_h = bias_a[h];
			float bias_h_eps = bias_h -eps;
			for(int s = 0; s < nSamples; ++s){
				incmplx = input_a[h][s];
				//norm = std::abs(incmplx);
				norm = sqrt((incmplx*conj(incmplx)).real());
				scale = 1.0f +bias_h_eps/(norm +eps);
				output_a[h][s] = (norm+bias_h >= 0) ? incmplx*scale : 0;
			}
		}
	});
	return output;
}
std::vector<at::Tensor> backwardmodReLU(
	torch::Tensor grad_output, torch::Tensor input, torch::Tensor bias)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto bias_a = bias.accessor<float,1>();
	const int nFeatures = grad_output_a.size(0);
	const int nSamples = grad_output_a.size(1);
	auto grad_input = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto grad_input_a = grad_input.accessor<c10::complex<float>,2>();
	auto grad_bias = torch::zeros(nFeatures,at::kFloat);
	auto grad_bias_a = grad_bias.accessor<float,1>();
	const float eps = 1e-5; // epsilon: Magic number

	at::parallel_for(0, nFeatures, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			float bias_h = bias_a[h];
			float bias_h_eps = bias_h -eps;
			float gbias = 0;
			float norm = 0, inv_norm_eps = 0;
			float factor = 0, scale = 0;
			c10::complex<float> incmplx = 0;
			c10::complex<float> gout = 0;
			for(int s = 0; s < nSamples; ++s){
				incmplx = input_a[h][s];
				//norm = std::abs(incmplx); 
				norm = sqrt((incmplx*conj(incmplx)).real()); 
				inv_norm_eps = 1.0f/(norm +eps);
				//factor = 0.5f*bias_h_eps*inv_norm_eps;
				//scale = 1.0f +factor*(1.0f +eps*inv_norm_eps) ;				
				scale = 1.0f +bias_h_eps*inv_norm_eps -0.5f*bias_h_eps*norm*inv_norm_eps*inv_norm_eps;
				gout = grad_output_a[h][s];
				if(norm +bias_h >= 0){
					grad_input_a[h][s] = scale*gout;
					gbias += 2.0f*inv_norm_eps*(gout*conj(incmplx)).real();
				}else{
					grad_input_a[h][s] = 0.0;
				}
			}
			grad_bias_a[h] = gbias;
		}
	});
	return {grad_input, grad_bias};
}

// (Complex) Diagonal unitary matrix 
torch::Tensor forwardCDiag1(
	torch::Tensor input, torch::Tensor omega)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto omega_a = omega.accessor<float,1>();
	const int nFeatures = input_a.size(0);
	const int nSamples = input_a.size(1);
	auto output = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto output_a = output.accessor<c10::complex<float>,2>();

	at::parallel_for(0, nFeatures, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			c10::complex<float> exp_iomega_h = exp(1._if*omega_a[h]);
			for(int s = 0; s < nSamples; ++s){
				output_a[h][s] = exp_iomega_h *input_a[h][s];
			}
		}
	});
	return output;
}
std::vector<at::Tensor> backwardCDiag1(
	torch::Tensor grad_output, torch::Tensor input, torch::Tensor omega)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto omega_a = omega.accessor<float,1>();
	const int nFeatures = grad_output_a.size(0);
	const int nSamples = grad_output_a.size(1);
	auto grad_input = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto grad_input_a = grad_input.accessor<c10::complex<float>,2>();
	auto grad_omega = torch::empty(nFeatures,at::kFloat);
	auto grad_omega_a = grad_omega.accessor<float,1>();

	at::parallel_for(0, nFeatures, 0, [&](int64_t start, int64_t end){
		for(int64_t h = start; h < end; ++h){
			c10::complex<float> exp_miomega_h = exp(-1._if*omega_a[h]);
			c10::complex<float> mi_exp_miomega_h = -1._if*exp_miomega_h;
			c10::complex<float> gout = 0;
			float gomega = 0;
			for(int s = 0; s < nSamples; ++s){
				gout = grad_output_a[h][s];
				grad_input_a[h][s] = exp_miomega_h*gout;
				gomega += 2.0*(mi_exp_miomega_h*conj(input_a[h][s])*gout).real();
			}
			grad_omega_a[h] = gomega;
		}
	});
	return {grad_input, grad_omega};
}

// Complex linear 
torch::Tensor forwardCLinear(
	torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto weight_a = weight.accessor<c10::complex<float>,2>();
	const auto bias_a = bias.accessor<c10::complex<float>,1>();
	const int nOutFeatures = weight_a.size(0);
	const int nInFeatures = weight_a.size(1);
	const int nSamples = input_a.size(1);
	auto output = torch::empty({nOutFeatures, nSamples}, c10::kComplexFloat);
	auto output_a = output.accessor<c10::complex<float>,2>();
	at::parallel_for(0, nOutFeatures, 0, [&](int64_t start, int64_t end)
	{
		for(int64_t o = start; o < end; ++o){
			auto output_o = output_a[o];
			for(int s = 0; s < nSamples; ++s){
				c10::complex<float> val = 0;
				for(int i = 0; i < nInFeatures; ++i)
					val += weight_a[o][i]*input_a[i][s];
				output_o[s] = bias_a[o] +val;
			}
		}
	});
	return output;
}
std::vector<at::Tensor> backwardCLinear(
	torch::Tensor grad_output, torch::Tensor input,
	torch::Tensor weight, torch::Tensor bias)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto weight_a = weight.accessor<c10::complex<float>,2>();
	const int nSamples = input_a.size(1);
	const int nInFeatures = input_a.size(0);
	const int nOutFeatures = grad_output_a.size(0);
	auto grad_input = torch::empty({nInFeatures, nSamples}, c10::kComplexFloat);
	auto grad_input_a = grad_input.accessor<c10::complex<float>,2>();
	auto grad_weight = torch::empty({nOutFeatures, nInFeatures}, c10::kComplexFloat);
	auto grad_weight_a = grad_weight.accessor<c10::complex<float>,2>();
	auto grad_bias = torch::empty(nOutFeatures, c10::kComplexFloat);
	auto grad_bias_a = grad_bias.accessor<c10::complex<float>,1>();

	at::parallel_for(0, nInFeatures, 0, [&](int64_t start, int64_t end)
	{
		for(int64_t i = start; i < end; ++i){
			auto grad_input_i = grad_input_a[i];
			for(int s = 0; s < nSamples; ++s){
				c10::complex<float> val = 0;
				for(int o = 0; o < nOutFeatures; ++o)
					val += conj(weight_a[o][i])*grad_output_a[o][s];
				grad_input_i[s] = val;
			}
		}
	});
	at::parallel_for(0, nOutFeatures, 0, [&](int64_t start, int64_t end)
	{
		for(int64_t o = start; o < end; ++o){
			for(int i = 0; i < nInFeatures; ++i){
				c10::complex<float> w = 0;
				for(int s = 0; s < nSamples; ++s)
					w += grad_output_a[o][s]*conj(input_a[i][s]);
				grad_weight_a[o][i] = w;
			}
			c10::complex<float> b = 0;
			for(int s = 0; s < nSamples; ++s)
				b += grad_output_a[o][s];
			grad_bias_a[o] = b;
		}
	});
	return {grad_input, grad_weight, grad_bias};
}


// Complex add
torch::Tensor forwardCadd(torch::Tensor z0, torch::Tensor z1)
{
	const auto z0_a = z0.accessor<c10::complex<float>,2>();
	const auto z1_a = z1.accessor<c10::complex<float>,2>();
	const int nFeatures = z0_a.size(0); 
	const int nSamples = z0_a.size(1); // Batch size
	auto output = torch::empty({nFeatures,nSamples},c10::kComplexFloat);
	auto output_a = output.accessor<c10::complex<float>,2>();

	at::parallel_for(0, nFeatures, 0, [&](int64_t start, int64_t end)
	{
		for(int64_t i = start; i < end; ++i){
			auto z0_i = z0_a[i];
			auto z1_i = z1_a[i];
			auto output_i = output_a[i];
			for(int s = 0; s < nSamples; ++s)
				output_i[s] = cmplxadd(z0_i[s],z1_i[s]);
		}
	});
	return output;
}
std::vector<at::Tensor> backwardCadd( torch::Tensor grad_output)
{
	using namespace c10::complex_literals;
	using namespace c10_complex_math;

	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const int nFeatures = grad_output_a.size(0);
	const int nSamples = grad_output_a.size(1);
	auto grad_z0 = torch::empty({nFeatures, nSamples}, c10::kComplexFloat);
	auto grad_z0_a = grad_z0.accessor<c10::complex<float>,2>();
	auto grad_z1 = torch::empty({nFeatures, nSamples}, c10::kComplexFloat);
	auto grad_z1_a = grad_z1.accessor<c10::complex<float>,2>();

	at::parallel_for(0, nFeatures, 0, [&](int64_t start, int64_t end)
	{
		for(int64_t i = start; i < end; ++i){
			auto gout_i = grad_output_a[i];
			auto gz0_i = grad_z0_a[i];
			auto gz1_i = grad_z1_a[i];
			for(int s = 0; s < nSamples; ++s)
				gz0_i[s] = gz1_i[s] = gout_i[s];
		}
	});
	return {grad_z0, grad_z1};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("set_num_threads", &set_num_threads, "set_num_threads");
	m.def("forwardClementsPSDC", &forwardClementsPSDC, "forward Clements-Structure (PSDC)^2");
	m.def("backwardClementsPSDC", &backwardClementsPSDC, "backward Clements-Structure (PSDC)^2");
	m.def("forwardClementsDCPS", &forwardClementsDCPS, "forward Clements-Structure (DCPS)^2");
	m.def("backwardClementsDCPS", &backwardClementsDCPS, "backward Clements-Structure (DCPS)^2");
	m.def("forwardClementsDBLarm", &forwardClementsDBLarm, "forward Clements-Structure (DCPS)(PSDC)");
	m.def("backwardClementsDBLarm", &backwardClementsDBLarm, "backward Clements-Structure (DCPS)(PSDC)");
	m.def("forwardPSDC", &forwardPSDC, "forward PSDC for CDcpp");
	m.def("backwardPSDC", &backwardPSDC, "backwardPSDC for CDcpp");
	m.def("forwardmodReLU", &forwardmodReLU, "forward modReLU");
	m.def("backwardmodReLU", &backwardmodReLU, "backward modReLU");
	m.def("forwardCDiag1", &forwardCDiag1, "forward complex Diagonal whose norm=1");
	m.def("backwardCDiag1", &backwardCDiag1, "backward complex Diagonal whose norm=1");
	m.def("forwardCLinear", &forwardCLinear, "forward complex linear");
	m.def("backwardCLinear", &backwardCLinear, "backward complex linear");
	m.def("forwardCadd", &forwardCadd, "forward complex add");
	m.def("backwardCadd", &backwardCadd, "backward complex add");
}
