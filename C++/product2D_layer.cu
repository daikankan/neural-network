#include <vector>

#include "caffe/layers/product2D_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Product2DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* lweight = this->blobs_[0]->gpu_data();
  const Dtype* rweight = this->blobs_[1]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_gpu_set(top[0]->count(), (Dtype)0, top[0]->mutable_gpu_data()); // set 0
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < this->N_; ++n) {
    this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, lweight,
                           rweight, top_data + n * this->top_dim_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[2]->gpu_data();
      this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  // Gradient with respect to bias
  if (this->bias_term_ && this->param_propagate_down_[2]) {
    Dtype* bias_diff = this->blobs_[2]->mutable_gpu_diff();
    for (int n = 0; n < this->N_; ++n) {
      this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
    }
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_gpu_set(bottom[0]->count(), (Dtype)0, bottom[0]->mutable_gpu_diff()); //set 0
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* lweight = this->blobs_[0]->gpu_data();
  const Dtype* rweight = this->blobs_[1]->gpu_data();
  // Gradient with respect to the left and right projection weight
  if (this->param_propagate_down_[0] && this->param_propagate_down_[1]) {
    Dtype* lweight_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* rweight_diff = this->blobs_[1]->mutable_gpu_diff();
    for (int n = 0; n < this->N_; ++n) {
      this->backward_gpu_weight(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_,
                                lweight, lweight_diff,
                                rweight, rweight_diff);
    }
  }
  // Gradient with respect to bottom data
  if (propagate_down[0]) {
    for (int n = 0; n < this->N_; ++n) {
      this->backward_gpu_gemm(top_diff + n * this-> top_dim_, lweight,
                              rweight, bottom_diff + n * this->bottom_dim_);
    }
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::forward_gpu_gemm(const Dtype* input, const Dtype* lweight,
                                             const Dtype* rweight, Dtype* output) {
  for (int c = 0; c < this->num_output_; ++c) {
    for (int i = 0; i < this->C_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, lweight_h_,
                            W_, H_, (Dtype)1.,
                            lweight + c * lweight_CHW_dim_ + i * lweight_spatial_dim_,
                            input + i * in_spatial_dim_,
                            (Dtype)0., rweight_multiplier_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, lweight_h_, rweight_w_, W_,
                            (Dtype)1., rweight_multiplier_.gpu_data(),
                            rweight + c * rweight_CHW_dim_ + i * rweight_spatial_dim_,
                            (Dtype)1., output + c * out_spatial_dim_);
    }
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        out_spatial_dim_, 1, (Dtype)1., bias,
                        bias_multiplier_.gpu_data(), (Dtype)1., output);
}

template <typename Dtype>
void Product2DLayer<Dtype>::backward_gpu_weight(const Dtype* output, const Dtype* input,
                                                const Dtype* lweight, Dtype* lweight_diff,
                                                const Dtype* rweight, Dtype* rweight_diff) {
  for (int c = 0; c < this->num_output_; ++c) {
    for (int i = 0; i < this->C_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            this->H_, this->rweight_w_, this->W_, (Dtype)1.,
                            output + i * in_spatial_dim_,
                            rweight + c * rweight_CHW_dim_ + i * rweight_spatial_dim_,
                            (Dtype)0., lweight_multiplier_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                            this->lweight_h_, this->H_, this->rweight_w_, (Dtype)1.,
                            input + c * out_spatial_dim_,
                            lweight_multiplier_.gpu_data(), (Dtype)1.,
                            lweight_diff + c * lweight_CHW_dim_ + i * lweight_spatial_dim_);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            this->lweight_h_, this->W_, this->H_, (Dtype)1.,
                            lweight + c * lweight_CHW_dim_ + i * lweight_spatial_dim_,
                            output + i * in_spatial_dim_,
                            (Dtype)0., rweight_multiplier_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            this->W_, this->rweight_w_, this->lweight_h_, (Dtype)1.,
                            rweight_multiplier_.gpu_data(),
                            input + c * out_spatial_dim_, (Dtype)1.,
                            rweight_diff + c * rweight_CHW_dim_ + i * rweight_spatial_dim_);
    }
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::backward_gpu_bias(Dtype* bias, const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                        input, bias_multiplier_.gpu_data(), 1., bias);
}

template <typename Dtype>
void Product2DLayer<Dtype>::backward_gpu_gemm(const Dtype* input, const Dtype* lweight,
                                              const Dtype* rweight, Dtype* output) {
  for (int i = 0; i < this->C_; ++i) {
    for (int c = 0; c < this->num_output_; ++c) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            this->H_, this->rweight_w_, this->lweight_h_, (Dtype)1.,
                            lweight + c * lweight_CHW_dim_ + i * lweight_spatial_dim_,
                            input + c * out_spatial_dim_, (Dtype)0.,
                            lweight_multiplier_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                            this->H_, this->W_, this->rweight_w_, (Dtype)1.,
                            lweight_multiplier_.gpu_data(),
                            rweight + c * rweight_CHW_dim_ + i * rweight_spatial_dim_,
                            (Dtype)1., output + i * in_spatial_dim_);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Product2DLayer);

}  // namespace caffe