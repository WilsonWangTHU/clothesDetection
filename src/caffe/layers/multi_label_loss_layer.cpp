// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "MULTI_LABEL_LOSS layer inputs must have the same count.";
  // if (top.size() >= 1) {
  //  // sigmoid cross entropy loss (averaged across batch)
  //   top[0]->Reshape(1, 1, 1, 1);
  // }
  // if (top.size() == 2) {
  //  // softmax output
  //   top[1]->ReshapeLike(*sigmoid_output_.get());
  //   top[1]->ShareData(*sigmoid_output_.get());
  // }
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    if (target[i] != 0) {
    // Update the loss only if target[i] is not 0
      loss -= input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    }
  }
  // if (top.size() >= 1) {
  //   top[0]->mutable_cpu_data()[0] = loss / num;
  // }
  top[0]->mutable_cpu_data()[0] = loss / num;
  //exit(1);
    // Scale down gradient
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    for (int i = 0; i < count; ++i) {
      
      if (target[i] != 0) {
        bottom_diff[i] = sigmoid_output_data[i] - (target[i] > 0);
      } else {
        bottom_diff[i] = 0;
      }
    }
    caffe_scal(count, loss_weight / num, bottom_diff);

  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiLabelLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelLossLayer);
REGISTER_LAYER_CLASS(MultiLabelLoss);

}  // namespace caffe
