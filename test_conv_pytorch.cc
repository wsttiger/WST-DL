#include <cassert>
#include <fstream>
#include "wstdl.h"

void test_conv(const std::string& inputFile, 
               const std::string& outputFile, 
               const std::string& kernelFile) {
  int batch_size, n_input_channels, size_h, size_w, n_output_channels;
  int f_size_h, f_size_w, stride_h, stride_w, padding_h, padding_w;

  std::ifstream fstr_kernel(kernelFile);
  std::vector<float> kernel;
  if (fstr_kernel.is_open()) {
    int nInChannels;
    fstr_kernel >> nInChannels;
    n_input_channels = nInChannels;
    int nOutChannels;
    fstr_kernel >> nOutChannels;
    n_output_channels = nOutChannels;
    int fH;
    fstr_kernel >> fH;
    f_size_h = fH;
    int fW;
    fstr_kernel >> fW;
    f_size_w = fW;
    int sz = nInChannels*nOutChannels*fH*fW; 
    int strideH;
    fstr_kernel >> strideH;
    stride_h = strideH;
    int strideW;
    fstr_kernel >> strideW;
    stride_w = strideW;
    int paddingH;
    fstr_kernel >> paddingH;
    padding_h = paddingH;
    int paddingW;
    fstr_kernel >> paddingW;
    padding_w = paddingW;
    kernel.resize(sz, 0.0f); 

    printf("Kernel:\n");
    printf("n_input_channels: %d\n", n_input_channels);
    printf("n_output_channels: %d\n", n_output_channels);
    printf("f_size_h: %d\n", f_size_h);
    printf("f_size_w: %d\n", f_size_w);
    printf("stride_h: %d\n", stride_h);
    printf("stride_w: %d\n", stride_w);
    for (auto& x : kernel) {
      fstr_kernel >> x;
      printf("%10.5e  ", x);
    }
    printf("\n\n");
  } 

  std::ifstream fstr_in(inputFile);
  std::vector<float> input;
  assert(fstr_in.is_open());
  {
    int nBatch;
    fstr_in >> nBatch;
    batch_size = nBatch;
    int nChannels;
    fstr_in >> nChannels;
    int nH;
    fstr_in >> nH;
    size_h = nH;
    int nW;
    fstr_in >> nW;
    size_w = nW;
    int sz = nBatch*nChannels*nH*nW; 
    input.resize(sz, 0.0f); 
    for (auto& x : input) {
      fstr_in >> x;
    }
  } 

  std::vector<float> output;
  int m_output = (size_h - f_size_h + 2*padding_h)/stride_h + 1;
  int n_output = (size_w - f_size_w + 2*padding_w)/stride_w + 1;
  output.resize(batch_size*n_output_channels*m_output*n_output);
  std::ifstream fstr_out(outputFile);
  std::vector<float> output_gold;
  assert(fstr_out.is_open());
  {
    int nBatch;
    fstr_out >> nBatch;
    int nChannels;
    fstr_out >> nChannels;
    int nH;
    fstr_out >> nH;
    int nW;
    fstr_out >> nW;
    int sz = nBatch*nChannels*nH*nW; 
    output_gold.resize(sz, 0.0f); 
    for (auto& x : output_gold) {
      fstr_out >> x;
    }
  } 

  conv2d_batch(batch_size, 
               n_input_channels, 
               size_h,
               size_w, 
               n_output_channels, 
               f_size_h,
               f_size_w,
               stride_h,
               stride_w,
               padding_h, 
               padding_w, 
               kernel.data(), input.data(), output.data());

  printf("\n\n");
  for (auto x : output) {
    printf("%15.10e  " , x); 
  } 
}

int main() {
  test_conv("input.dat", "output.dat", "conv2d.dat");    
  return 0;
}
