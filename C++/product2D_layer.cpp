#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/product2D_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Product2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  N_ = bottom[0]->shape(-4);
  C_ = bottom[0]->shape(-3);
  H_ = bottom[0]->shape(-2);
  W_ = bottom[0]->shape(-1);
  num_output_ = this->layer_param_.product2d_param().num_output();
  lweight_h_ = this->layer_param_.product2d_param().lweight_h();
  rweight_w_ = this->layer_param_.product2d_param().rweight_w();
  bias_term_ = this->layer_param_.product2d_param().bias_term();
  // Handle the parameters: left, right weights and biases.
  // For example, if bottom[0]'s shape is (N_, C_, H_, W_)
  // the shape of left weights should be (num_output_, C_, lweight_h_, H_)
  // the shape of right weights should be (num_output_, C_, W_, rweight_w_)
  // i.e. top[0]'s shape is (N_, num_output_, lwe`ight_h_, rweight_w_)
  vector<int> lweight_shape(4);
  lweight_shape[0] = num_output_;
  lweight_shape[1] = bottom[0]->shape(1);
  lweight_shape[2] = lweight_h_;
  lweight_shape[3] = bottom[0]->shape(2);
  vector<int> rweight_shape(4);
  rweight_shape[0] = num_output_;
  rweight_shape[1] = bottom[0]->shape(1);
  rweight_shape[2] = bottom[0]->shape(3);
  rweight_shape[3] = rweight_w_;
  // - blobs_[0] holds the left weights
  // - blobs_[1] holds the right weights
  // - blobs_[2] holds the biases (optional)
  if (this->blobs_.size() > 0) {
    CHECK_EQ(3, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (lweight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> lweight_shaped_blob(lweight_shape);
      LOG(FATAL) << "Incorrect left weight shape: expected shape "
                 << lweight_shaped_blob.shape_string()
                 << "; instead, shape was "
                 << this->blobs_[0]->shape_string();
    }
    if (rweight_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> rweight_shaped_blob(rweight_shape);
      LOG(FATAL) << "Incorrect right weight shape: expected shape "
                 << rweight_shaped_blob.shape_string()
                 << "; instead, shape was "
                 << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initializaiton";
  } else {
    if (bias_term_) {
      this->blobs_.resize(3);
    } else {
      this->blobs_.resize(2);
    }
    // Initialize the left and right weights
    this->blobs_[0].reset(new Blob<Dtype>(lweight_shape));
    this->blobs_[1].reset(new Blob<Dtype>(rweight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.product2d_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    weight_filler->Fill(this->blobs_[1].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.product2d_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void Product2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  // top[0]'s shape is (N_, num_output_, lweight_h_, rweight_w_)
  CHECK_EQ(bottom[0]->shape(0), N_)
      << "bottom size may not change.";
  vector<int> top_shape(4);
  top_shape[0] = N_;
  top_shape[1] = num_output_;
  top_shape[2] = lweight_h_;
  top_shape[3] = rweight_w_;
  top[0]->Reshape(top_shape);
  bottom_dim_ = bottom[0]->count(1);
  top_dim_ = top[0]->count(1);
  CHECK_EQ(bottom[0]->shape(3), W_) << "bottom shape W_ incorrect.";
  CHECK_EQ(bottom[0]->shape(2), H_) << "bottom shape H_ incorrect.";
  // Set up the "left right weight multiplier" for cross-gradient-iteration,
  // also used as intermediate results for forward calculation.
  vector<int> lweight_multiplier_shape(2);
  lweight_multiplier_shape[0] = H_;
  lweight_multiplier_shape[1] = rweight_w_;
  vector<int> rweight_multiplier_shape(2);
  rweight_multiplier_shape[0] = lweight_h_;
  rweight_multiplier_shape[1] = W_;
  lweight_multiplier_.Reshape(lweight_multiplier_shape);
  rweight_multiplier_.Reshape(rweight_multiplier_shape);
  lweight_CHW_dim_ = C_ * lweight_h_ * H_;
  lweight_spatial_dim_ = lweight_h_ * H_;
  rweight_CHW_dim_ = C_ * W_ * rweight_w_;
  rweight_spatial_dim_ = W_* rweight_w_;
  in_spatial_dim_ = bottom[0]->count(2);
  // Set up the all one "bias multiplier" for adding biases
  out_spatial_dim_ = top[0]->count(2);
  CHECK_EQ(out_spatial_dim_, lweight_h_*rweight_w_)
      << "top spatial dim incorrect.";
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
              bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* lweight = this->blobs_[0]->cpu_data();
  const Dtype* rweight = this->blobs_[1]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_set(top[0]->count(), (Dtype)0, top[0]->mutable_cpu_data()); // set 0
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < this->N_; ++n) {
    this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, lweight,
                           rweight, top_data + n * this->top_dim_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[2]->cpu_data();
      this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  // Gradient with respect to bias
  if (this->bias_term_ && this->param_propagate_down_[2]) {
    Dtype* bias_diff = this->blobs_[2]->mutable_cpu_diff();
    for (int n = 0; n < this->N_; ++n) {
      this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
    }
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_set(bottom[0]->count(), (Dtype)0, bottom[0]->mutable_cpu_diff()); //set 0
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* lweight = this->blobs_[0]->cpu_data();
  const Dtype* rweight = this->blobs_[1]->cpu_data();
  // Gradient with respect to the left and right projection weight
  if (this->param_propagate_down_[0] && this->param_propagate_down_[1]) {
    Dtype* lweight_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* rweight_diff = this->blobs_[1]->mutable_cpu_diff();
    for (int n = 0; n < this->N_; ++n) {
      this->backward_cpu_weight(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_,
                                lweight, lweight_diff,
                                rweight, rweight_diff);
    }
  }
  // Gradient with respect to bottom data
  if (propagate_down[0]) {
    for (int n = 0; n < this->N_; ++n) {
      this->backward_cpu_gemm(top_diff + n * this-> top_dim_, lweight,
                              rweight, bottom_diff + n * this->bottom_dim_);
    }
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* lweight,
                                             const Dtype* rweight, Dtype* output) {
  for (int c = 0; c < this->num_output_; ++c) {
    for (int i = 0; i < this->C_; ++i) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, lweight_h_,
                            W_, H_, (Dtype)1.,
                            lweight + c * lweight_CHW_dim_ + i * lweight_spatial_dim_,
                            input + i * in_spatial_dim_,
                            (Dtype)0., rweight_multiplier_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, lweight_h_, rweight_w_, W_,
                            (Dtype)1., rweight_multiplier_.cpu_data(),
                            rweight + c * rweight_CHW_dim_ + i * rweight_spatial_dim_,
                            (Dtype)1., output + c * out_spatial_dim_);
    }
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        out_spatial_dim_, 1, (Dtype)1., bias,
                        bias_multiplier_.cpu_data(), (Dtype)1., output);
}

template <typename Dtype>
void Product2DLayer<Dtype>::backward_cpu_weight(const Dtype* output, const Dtype* input,
                                                const Dtype* lweight, Dtype* lweight_diff,
                                                const Dtype* rweight, Dtype* rweight_diff) {
  for (int c = 0; c < this->num_output_; ++c) {
    for (int i = 0; i < this->C_; ++i) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            this->H_, this->rweight_w_, this->W_, (Dtype)1.,
                            output + i * in_spatial_dim_,
                            rweight + c * rweight_CHW_dim_ + i * rweight_spatial_dim_,
                            (Dtype)0., lweight_multiplier_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                            this->lweight_h_, this->H_, this->rweight_w_, (Dtype)1.,
                            input + c * out_spatial_dim_,
                            lweight_multiplier_.cpu_data(), (Dtype)1.,
                            lweight_diff + c * lweight_CHW_dim_ + i * lweight_spatial_dim_);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            this->lweight_h_, this->W_, this->H_, (Dtype)1.,
                            lweight + c * lweight_CHW_dim_ + i * lweight_spatial_dim_,
                            output + i * in_spatial_dim_,
                            (Dtype)0., rweight_multiplier_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            this->W_, this->rweight_w_, this->lweight_h_, (Dtype)1.,
                            rweight_multiplier_.cpu_data(),
                            input + c * out_spatial_dim_, (Dtype)1.,
                            rweight_diff + c * rweight_CHW_dim_ + i * rweight_spatial_dim_);
    }
  }
}

template <typename Dtype>
void Product2DLayer<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, (Dtype)1.,
                        input, bias_multiplier_.cpu_data(), (Dtype)1., bias);
}

template <typename Dtype>
void Product2DLayer<Dtype>::backward_cpu_gemm(const Dtype* input, const Dtype* lweight,
                                              const Dtype* rweight, Dtype* output) {
  for (int i = 0; i < this->C_; ++i) {
    for (int c = 0; c < this->num_output_; ++c) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            this->H_, this->rweight_w_, this->lweight_h_, (Dtype)1.,
                            lweight + c * lweight_CHW_dim_ + i * lweight_spatial_dim_,
                            input + c * out_spatial_dim_, (Dtype)0.,
                            lweight_multiplier_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                            this->H_, this->W_, this->rweight_w_, (Dtype)1.,
                            lweight_multiplier_.cpu_data(),
                            rweight + c * rweight_CHW_dim_ + i * rweight_spatial_dim_,
                            (Dtype)1., output + i * in_spatial_dim_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Product2DLayer);
#endif

INSTANTIATE_CLASS(Product2DLayer);
REGISTER_LAYER_CLASS(Product2D);

}  // namespace caffe
