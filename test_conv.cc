#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

#include "wstdl.h"

void test_conv2d_3x3_no_padding_channel_1_stride_1() {
  const int f[2] = {3, 3};
  const int size[2] = {4, 6};
  const int padding[2] = {0, 0};
  const int stride[2] = {1, 1};
  const int n_input_channels = 1;

  // input
  const int m_input = size[0];
  const int n_input = size[1];
  const int fsize = f[0]*f[1]*n_input_channels;
  std::vector<float> input(m_input*n_input);
  for (unsigned int i = 0; i < input.size(); i++) {
    input[i] = 0.031*i + std::cos(0.1434*i)*std::sin(1.964*i);
  }
  // kernel
  std::vector<float> kn(fsize);
  std::iota(kn.begin(), kn.end(), 1);
  // output
  const int m_output = (m_input - f[0] + 2*padding[0])/stride[0] + 1;
  const int n_output = (n_input - f[1] + 2*padding[1])/stride[1] + 1;
  std::vector<float> output(m_output*n_output);

  conv2d(m_input, n_input, f[0], f[1], n_input_channels, stride[0], stride[1], padding[0], padding[1], kn.data(), input.data(), output.data()); 

  std::vector<float> correct = {11.20000, 15.06938, 18.47373, 14.59331, 18.75166, 22.09858, 28.22423, 23.64482};

  float tol = 1e-4;
  for (auto i = 0; i < output.size(); i++) {
    assert(std::abs(output[i]-correct[i]) < tol);
  }
}

void test_conv2d_3x3_no_padding_channel_2_stride_1() {
  const int f[2] = {3, 3};
  const int size[2] = {4, 6};
  const int padding[2] = {0, 0};
  const int stride[2] = {1, 1};
  const int n_input_channels = 2;

  // input
  const int m_input = size[0];
  const int n_input = size[1];
  const int fsize = f[0]*f[1]*n_input_channels;
  std::vector<float> input(m_input*n_input*n_input_channels);
  for (unsigned int i = 0; i < m_input*n_input; i++) {
    input[i] = 0.031*i + std::cos(0.1434*i)*std::sin(1.964*i);
    input[i+m_input*n_input] = input[i];
  }
  // kernel is 2 channels with both channels containing the same sequence 
  // of values, i.e. {1,2,3,4,5,6,7,8,9}
  // results should be double that of test with single channel
  std::vector<float> kn(fsize);
  std::iota(kn.begin(), kn.begin()+9, 1);
  std::iota(kn.begin()+9, kn.end(), 1);
  // output
  const int m_output = (m_input - f[0] + 2*padding[0])/stride[0] + 1;
  const int n_output = (n_input - f[1] + 2*padding[1])/stride[1] + 1;
  std::vector<float> output(m_output*n_output);

  conv2d(m_input, n_input, f[0], f[1], n_input_channels, stride[0], stride[1], padding[0], padding[1], kn.data(), input.data(), output.data()); 

  std::vector<float> correct = {11.20000, 15.06938, 18.47373, 14.59331, 18.75166, 22.09858, 28.22423, 23.64482};

  float tol = 1e-4;
  for (auto i = 0; i < output.size(); i++) {
    assert(std::abs(output[i]-2*correct[i]) < tol);
  }
}

void test_conv2d_3x3_padding_1_channel_1_stride_1() {
  const int f[2] = {3, 3};
  const int size[2] = {4, 6};
  const int padding[2] = {1, 1};
  const int stride[2] = {1, 1};
  const int n_input_channels = 1;

  // input
  const int m_input = size[0];
  const int n_input = size[1];
  const int fsize = f[0]*f[1]*n_input_channels;
  std::vector<float> input(m_input*n_input);
  for (unsigned int i = 0; i < input.size(); i++) {
    input[i] = 0.031*i + std::cos(0.1434*i)*std::sin(1.964*i);
  }
  // kernel
  std::vector<float> kn(fsize);
  std::iota(kn.begin(), kn.end(), 1);
  // output
  const int m_output = (m_input - f[0] + 2*padding[0])/stride[0] + 1;
  const int n_output = (n_input - f[1] + 2*padding[1])/stride[1] + 1;
  std::vector<float> output(m_output*n_output);

  conv2d(m_input, n_input, f[0], f[1], n_input_channels, stride[0], stride[1], padding[0], padding[1], kn.data(), input.data(), output.data()); 

  //std::vector<float> correct = {11.20000, 15.06938, 18.47373, 14.59331, 18.75166, 22.09858, 28.22423, 23.64482};

  // float tol = 1e-4;
  // for (auto i = 0; i < output.size(); i++) {
  //   assert(std::abs(output[i]-correct[i]) < tol);
  // }

}

void test_im2col_3x3_no_pad_identity() {
  const int f[2] = {3, 3};
  const int size[2] = {3, 3};
  const int padding[2] = {0, 0};
  const int stride[2] = {1, 1};
  const int n_input_channels = 1;

  const int m_input = size[0];
  const int n_input = size[1];
  std::vector<int> input(m_input*n_input);
  std::iota(input.begin(), input.end(), 0);

  const int m_output = (m_input - f[0] + 2*padding[0])/stride[0] + 1;
  const int n_output = (n_input - f[1] + 2*padding[1])/stride[1] + 1;
  std::vector<int> col(m_output*n_output*f[0]*f[1]);

  im2col(m_input, n_input, f[0], f[1], n_input_channels, stride[0], stride[1], padding[0], padding[1], input.data(), col.data());

  for (auto i = 0; i < col.size(); i++) {
    assert(input[i]==col[i]);
  }
}

void test_im2col_3x3_no_pad() {
  const int f[2] = {3, 3};
  const int size[2] = {4, 4};
  const int padding[2] = {0, 0};
  const int stride[2] = {1, 1};
  const int n_input_channels = 1;

  const int m_input = size[0];
  const int n_input = size[1];
  std::vector<int> input(m_input*n_input);
  std::iota(input.begin(), input.end(), 0);

  int fsize = f[0]*f[1]*n_input_channels;
  const int m_output = (m_input - f[0] + 2*padding[0])/stride[0] + 1;
  const int n_output = (n_input - f[1] + 2*padding[1])/stride[1] + 1;
  std::vector<int> col(m_output*n_output*fsize);

  im2col(m_input, n_input, f[0], f[1], n_input_channels, stride[0], stride[1], padding[0], padding[1], input.data(), col.data());

  std::vector<int> correct = {0, 1, 2, 4, 5, 6, 8, 9, 10, 1, 2, 3, 5, 6, 7, 9, 10, 11, 
                              4, 5, 6, 8, 9, 10, 12, 13, 14, 5, 6, 7, 9, 10, 11, 13, 14, 15};

  for (auto i = 0; i < col.size(); i++) {
    assert(correct[i]==col[i]);
  }
}

void test_im2col_3x3_with_channels_identity() {
  const int f[2] = {3, 3};
  const int size[2] = {3, 3};
  const int padding[2] = {0, 0};
  const int stride[2] = {1, 1};
  const int n_input_channels = 2;

  const int m_input = size[0];
  const int n_input = size[1];
  std::vector<int> input(m_input*n_input*n_input_channels);
  std::iota(input.begin(), input.end(), 0);

  int fsize = f[0]*f[1]*n_input_channels;
  const int m_output = (m_input - f[0] + 2*padding[0])/stride[0] + 1;
  const int n_output = (n_input - f[1] + 2*padding[1])/stride[1] + 1;
  std::vector<int> col(m_output*n_output*fsize);

  im2col(m_input, n_input, f[0], f[1], n_input_channels, stride[0], stride[1], padding[0], padding[1], input.data(), col.data());

  for (auto i = 0; i < col.size(); i++) {
    assert(input[i]==col[i]);
  }
}

void test_im2col_3x3_with_padding() {
  const int f[2] = {3, 3};
  const int size[2] = {2, 2};
  const int padding[2] = {1, 1};
  const int stride[2] = {1, 1};
  const int n_input_channels = 1;

  const int m_input = size[0];
  const int n_input = size[1];
  std::vector<int> input(m_input*n_input);
  std::iota(input.begin(), input.end(), 0);

  int fsize = f[0]*f[1]*n_input_channels;
  const int m_output = (m_input - f[0] + 2*padding[0])/stride[0] + 1;
  const int n_output = (n_input - f[1] + 2*padding[1])/stride[1] + 1;
  std::vector<int> col(m_output*n_output*fsize);

  im2col(m_input, n_input, f[0], f[1], n_input_channels, stride[0], stride[1], padding[0], padding[1], input.data(), col.data());

  std::vector<int> correct = {0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0, 
                              0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0};

  for (auto i = 0; i < col.size(); i++) {
    assert(correct[i]==col[i]);
  }
}

int main() {

  test_im2col_3x3_no_pad_identity();
  test_im2col_3x3_no_pad();
  test_im2col_3x3_with_padding();
  test_im2col_3x3_with_channels_identity();
  test_conv2d_3x3_no_padding_channel_1_stride_1();
  test_conv2d_3x3_padding_1_channel_1_stride_1();
  test_conv2d_3x3_no_padding_channel_2_stride_1();

  return 0;
}
