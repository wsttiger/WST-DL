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
  printf("Print matrix: m -> %d    n -> %d\n", m, n);
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
            int dilation_h,
            int dilation_w,
            T* in, T* out) {
  int fsize = f_size_h*f_size_w*n_input_channels;
  int fdiv2_h, fdiv2_w;
  fdiv2_h = f_size_h/2; fdiv2_w = f_size_w/2; 
  // calculate output sizes
  const int mm = (size_h + 2*padding_h - dilation_h * (f_size_h - 1) - 1)/stride_h + 1;
  const int nn = (size_h + 2*padding_w - dilation_w * (f_size_w - 1) - 1)/stride_w + 1;
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
            int idx = ii + dilation_h*k;   
            int jdx = jj + dilation_w*l;
            if ((idx >= 0) && (idx < size_h) && 
                (jdx >= 0) && (jdx < size_w)) {
              out[ldx*fsize+p] = in[c*npixels+idx*size_w+jdx];
            }
            else {
              out[ldx*fsize+p] = T(0);
            } 
          }
        }
      }
    }
  }  
}

// NOTE:: This actually computes the cross correlation and not a strict convolution
void conv2d(int n_input_channels,
            int size_h, 
            int size_w, 
            int n_output_channels, 
            int f_size_h, 
            int f_size_w, 
            int stride_h, 
            int stride_w, 
            int padding_h, 
            int padding_w, 
            int dilation_h, 
            int dilation_w, 
            float* kn, float* in, float* out) {
  int fsize = f_size_h*f_size_w*n_input_channels;
  const int m_output = (size_h + 2*padding_h - dilation_h * (f_size_h - 1) - 1)/stride_h + 1;
  const int n_output = (size_h + 2*padding_w - dilation_w * (f_size_w - 1) - 1)/stride_w + 1;
  int mm = m_output*n_output;
  int kk = fsize;
  std::vector<float> col(mm*kk);

  im2col(size_h, size_w, 
        f_size_h, f_size_w, 
        n_input_channels, 
        stride_h, stride_w, 
        padding_h, padding_w, 
        dilation_h, dilation_w, 
        in, col.data());

  std::vector<float> out_tmp(n_output_channels*m_output*n_output);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
  mm, n_output_channels, kk, 1.0, col.data(), kk, kn, kk, 0.0, out_tmp.data(), n_output_channels);  
  int tmpM = m_output*n_output;
  int tmpN = n_output_channels;
  for (int i = 0; i < tmpM; i++) {
    for (int j = 0; j <tmpN; j++) {
      out[j*tmpM+i] = out_tmp[i*tmpN+j];
    }
  }
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
                  int dilation_h,
                  int dilation_w,
                  float* kn, float* in, float* out) {
  const int fsize = f_size_h*f_size_w*n_input_channels;
  const int output_size_h = (size_h + 2*padding_h - dilation_h * (f_size_h - 1) - 1)/stride_h + 1;
  const int output_size_w = (size_h + 2*padding_w - dilation_w * (f_size_w - 1) - 1)/stride_w + 1;

  const int input_frame_size = n_input_channels*size_h*size_w;
  const int output_frame_size = n_output_channels*output_size_h*output_size_w;
  for (int b = 0; b < batch_size; b++) {
    conv2d(n_input_channels, size_h, size_w, n_output_channels, 
           f_size_h, f_size_w, stride_h, stride_w, 
           padding_h, padding_w, dilation_h, dilation_w, kn,  
           &in[b*input_frame_size], &out[b*output_frame_size]);
  }
}

