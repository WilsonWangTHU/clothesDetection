/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"

#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
HDF5GeneralDataLayer<Dtype>::~HDF5GeneralDataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5GeneralDataLayer<Dtype>::LoadGeneralHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading The general HDF5 file" << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(ERROR) << "Failed opening HDF5 file" << filename;
  }

  HDF5GeneralDataParameter data_param = this->layer_param_.hdf5_general_data_param();
  int fieldNum = data_param.field_size();
  hdf_blobs_.resize(fieldNum);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = 4;
  for(int i = 0; i < fieldNum; ++i){
	  //LOG(INFO) << "Data type: " << data_param.datatype(i).data();
	  if(i < data_param.datatype_size() && 
      strcmp(data_param.datatype(i).data(), "int8") == 0){
		  
      // We take out the io functions here
      const char* dataset_name_ = data_param.field(i).data();
      hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());

		  CHECK(H5LTfind_dataset(file_id, dataset_name_))
        << "Failed to find HDF5 dataset " << dataset_name_;
      // Verify that the number of dimensions is in the accepted range.
      herr_t status;
      int ndims;
      status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
      CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
      CHECK_GE(ndims, MIN_DATA_DIM);
      CHECK_LE(ndims, MAX_DATA_DIM);
      
      // Verify that the data format is what we expect: int8
      std::vector<hsize_t> dims(ndims);
      H5T_class_t class_;
      status = H5LTget_dataset_info(file_id, dataset_name_, dims.data(), &class_, NULL);
      CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
      CHECK_EQ(class_, H5T_INTEGER) << "Expected integer data";

      vector<int> blob_dims(dims.size());
      for (int j = 0; j < dims.size(); ++j) {
        blob_dims[j] = dims[j];
      }
      hdf_blobs_[i]->Reshape(blob_dims);
      std::cout<<"Trying to allocate memories!\n";
		  int* buffer_data = new int[hdf_blobs_[i]->count()];
      std::cout<<"Memories loaded!!!\n";
		  status = H5LTread_dataset_int(file_id, dataset_name_, buffer_data);
		  CHECK_GE(status, 0) << "Failed to read int8 dataset " << dataset_name_;

		  Dtype* target_data = hdf_blobs_[i]->mutable_cpu_data();
		  for(int j = 0; j < hdf_blobs_[i]->count(); j++){
			  //LOG(INFO) << Dtype(buffer_data[j]);
			  target_data[j] = Dtype(buffer_data[j]);
		  }
		  delete buffer_data;

	  }else{
      // The dataset is still the float32 datatype
      hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
		  hdf5_load_nd_dataset(file_id, data_param.field(i).data(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
	  }
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file " << filename;

  for(int i = 1; i < fieldNum; ++i){
	  CHECK_EQ(hdf_blobs_[0]->num(), hdf_blobs_[i]->num());
  }
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;
  //TODO: DATA SHUFFLE
  //LOG(INFO) << "Successully loaded " << data_blob_.num() << " rows";
}


template <typename Dtype>
void HDF5GeneralDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  HDF5GeneralDataParameter data_param = this->layer_param_.hdf5_general_data_param();
  // Read the source to parse the filenames.
  const string& source = data_param.source();
  LOG(INFO) << "Loading filename from " << source;
  int fieldNum = data_param.field_size();
  LOG(INFO) << "Number of fields in batches: " << fieldNum;

  hdf_blobs_.clear();
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of files: " << num_files_;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);  
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  // TODO: The shuffle is unable currently

  // Load the first HDF5 file and initialize the line counter.
  LoadGeneralHDF5FileData(hdf_filenames_[current_file_].c_str());

  current_row_ = 0;
  // Reshape blobs.
  const int batch_size = data_param.batch_size();
  const int top_size = this->layer_param_.top_size();
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
    top_shape.resize(hdf_blobs_[i]->num_axes());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_[i]->shape(j);
    }
    top[i]->Reshape(top_shape);
  }

}

template <typename Dtype>
void HDF5GeneralDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_general_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == hdf_blobs_[0]->shape(0)) {
      if (num_files_ > 1) {
        ++current_file_;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          //if (this->layer_param_.hdf5_data_param().shuffle()) {
          //  std::random_shuffle(file_permutation_.begin(),
          //                      file_permutation_.end());
          //}
          DLOG(INFO) << "Looping around to first file.";
        }
        LoadGeneralHDF5FileData(
          hdf_filenames_[file_permutation_[current_file_]].c_str());
      }
      current_row_ = 0;
      // TODO: SHUFFLE
      //if (this->layer_param_.hdf5_general_data_param().shuffle())
      //  std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5GeneralDataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5GeneralDataLayer);
REGISTER_LAYER_CLASS(HDF5GeneralData);
//REGISTER_LAYER_CLASS(HDF5_GENERAL_DATA);

}  // namespace caffe
