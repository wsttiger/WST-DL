#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <numeric>
#include <mkl.h>

void print(int n_input_channels, int m, int n, float* data) {
  for (int c = 0; c < n_input_channels; c++) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        printf("%10.7f   ", data[i*n+j]);
      }
      printf("\n");
    }
    printf("\n");
    printf("\n");
  }
}

void print_matrix(int m, int n, float* data) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%10.7f   ", data[i*n+j]);
    }
    printf("\n");
  }
  printf("\n");
}

template <typename T>
void im2col(int size_h, 
            int size_w, 
            int f_size_h, 
            int f_size_w, 
            int n_input_channels, 
            int stride_h, 
            int stride_w, 
            int padding_h, 
            int padding_w,
            T* in, T* out) {
  int fsize = f_size_h*f_size_w*n_input_channels;
  int fdiv2_h, fdiv2_w;
  fdiv2_h = f_size_h/2; fdiv2_w = f_size_w/2; 
  // calculate output sizes
  const int mm = (size_h - f_size_h + 2*padding_h)/stride_h + 1;
  const int nn = (size_w - f_size_w + 2*padding_w)/stride_w + 1;
  // calculate number of pixels in frame
  const int npixels = size_h*size_w;
  // starting pixel positions
  int ii = fdiv2_h - padding_h;
  // loop over output pixels
  for (int io = 0; io < mm; io++, ii+=stride_h) {
    int jj = fdiv2_w - padding_w;
    for (int jo = 0; jo < nn; jo++, jj+=stride_w) {
      // compute linear index for output
      int ldx = io*nn+jo;  
      int p = 0;
      for (int c = 0; c < n_input_channels; c++) {
        for (int k = -fdiv2_h; k <= fdiv2_h; k++) {
          for (int l = -fdiv2_w; l <= fdiv2_w; l++, p++) {
            int idx = ii + k;   
            int jdx = jj + l;
            if ((idx >= 0) && (idx < size_h) && 
                (jdx >= 0) && (jdx < size_w)) {
              out[ldx*fsize+p] = in[c*npixels+idx*size_w+jdx];
            }
          }
        }
      }
    }
  }  
}

// NOTE:: This actually computes the cross correlation and not a strict convolution
void conv2d(int size_h, 
            int size_w, 
            int f_size_h, 
            int f_size_w, 
            int n_input_channels, 
            int stride_h, 
            int stride_w, 
            int padding_h, 
            int padding_w, 
            float* kn, float* in, float* out) {
  int fsize = f_size_h*f_size_w*n_input_channels;
  const int m_output = (size_h - f_size_h + 2*padding_h)/stride_h + 1;
  const int n_output = (size_w - f_size_w + 2*padding_w)/stride_w + 1;
  int mm = m_output*n_output;
  int kk = fsize;
  std::vector<float> col(mm*kk);

  im2col(size_h, size_w, f_size_h, f_size_w, n_input_channels, stride_h, stride_w, padding_h, padding_w, in, col.data());

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
  mm, 1, kk, 1.0, col.data(), kk, kn, kk, 0.0, out, 1);  
}

// No dilation for now
// kn -> (n_output_channels) X (n_input_channels) X (kernel_height) X (kernel_width)
// in -> (batch_size) X (n_input_channels) X (input_height) X (input_width)
// out -> (batch_size) X (n_output_channels) X (output_height) X (output_width)
void conv2d_batch(int batch_size, 
                  int n_input_channels, 
                  int size_h,
                  int size_w, 
                  int n_output_channels, 
                  int f_size_h,
                  int f_size_w,
                  int stride_h,
                  int stride_w,
                  int padding_h, 
                  int padding_w, 
                  float* kn, float* in, float* out) {
  const int fsize = f_size_h*f_size_w*n_input_channels;
  const int output_size_h = (size_h - f_size_h + 2*padding_h)/stride_h + 1;
  const int output_size_w = (size_w - f_size_w + 2*padding_w)/stride_w + 1;

  const int input_kernel_size = n_input_channels*f_size_h*f_size_w;
  const int input_frame_size = size_h*size_w;
  const int input_total_size = n_input_channels*input_frame_size;
  const int output_frame_size = output_size_h*output_size_w;
  const int output_total_size = n_output_channels*output_frame_size;
    for (int b = 0; b < batch_size; b++) {
      for (int c = 0; c < n_output_channels; c++) {
        conv2d(size_h, size_w, f_size_h, f_size_w, 
               n_input_channels, stride_h, stride_w, 
               padding_h, padding_w, &kn[c*input_kernel_size],  
               &in[b*input_total_size+c*input_frame_size], &out[b*output_total_size+c*output_frame_size]);
    }
  }
}

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
