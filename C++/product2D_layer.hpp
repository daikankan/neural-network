#ifndef CAFFE_PRODUCT2D_LAYER_HPP_
#define CAFFE_PRODUCT2D_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief New layer, computes an product with a set of left and right weights,
 *        and optionally adds biases.
 *
 * TODD(dox): thorough documentation for forward, backward, and proto params.
 */
template <typename Dtype>
class Product2DLayer : public Layer<Dtype> {
 public:
  explicit Product2DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Product2D"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  void forward_cpu_gemm(const Dtype* input, const Dtype* lweight,
                        const Dtype* rweight, Dtype* output);
  void forward_gpu_gemm(const Dtype* input, const Dtype* lweight,
                        const Dtype* rweight, Dtype* output);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);

  void backward_cpu_weight(const Dtype* output, const Dtype* input,
                           const Dtype* lweight, Dtype* lweight_diff,
                           const Dtype* rweight, Dtype* rweight_diff);
  void backward_gpu_weight(const Dtype* output, const Dtype* input,
                           const Dtype* lweight, Dtype* lweight_diff,
                           const Dtype* rweight, Dtype* rweight_diff);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
  void backward_cpu_gemm(const Dtype* input, const Dtype* lweight,
                         const Dtype* rweight, Dtype* output);
  void backward_gpu_gemm(const Dtype* input, const Dtype* lweight,
                         const Dtype* rweight, Dtype* output);

  int N_;
  int C_;
  int H_;
  int W_;
  int num_output_;
  int lweight_h_;
  int rweight_w_;
  int in_spatial_dim_;
  int out_spatial_dim_;
  int lweight_CHW_dim_;
  int lweight_spatial_dim_;
  int rweight_CHW_dim_;
  int rweight_spatial_dim_;
  int bottom_dim_;
  int top_dim_;
  bool bias_term_;

  Blob<Dtype> lweight_multiplier_;
  Blob<Dtype> rweight_multiplier_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_PRODUCT2D_LAYER_HPP_
