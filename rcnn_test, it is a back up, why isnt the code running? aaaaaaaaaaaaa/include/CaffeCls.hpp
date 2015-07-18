#ifndef CAFFE_CLASS
#define CAFFE_CLASS

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "structureEdge.h"
#include "person_craft.h"
using namespace caffe;
using namespace std;
using namespace cv;

struct Bbox {
	float rootcoord[4];
	float score;
};

struct PropBox {
	float x1, y1, x2, y2;
	float confidence;
	int cls_id;
};

//------------- Base Class ---------------
class CaffeCls {
public:
	CaffeCls()
	{
		caffe_test_net_ = NULL;
	}
	void set_model(string proto_name, string model_name){
		Caffe::set_mode(Caffe::CPU);

		//initial test net
		NetParameter test_net_param;
		ReadNetParamsFromTextFileOrDie(proto_name, &test_net_param);
		caffe_test_net_ = new Net<float>(test_net_param);
		printf("init test net proto ok\n");

		//read in pretrained net
		NetParameter trained_net_param;
		ReadNetParamsFromBinaryFileOrDie(model_name, &trained_net_param);
		caffe_test_net_->CopyTrainedLayersFrom(trained_net_param);
		printf("init test net prarameters ok\n");
	}

	void set_means(float b, float g, float r){
		pixel_means_[0] = b;
		pixel_means_[1] = g;
		pixel_means_[2] = r;
	}

	virtual ~CaffeCls() {
		if (caffe_test_net_ != NULL)
			delete caffe_test_net_;
	};

protected:
	Net<float>* caffe_test_net_;
	float pixel_means_[3];
};


//-------------------- FastRCNN Class -----------------------
class FastRCNNCls: virtual public CaffeCls {
public:
	FastRCNNCls();
	~FastRCNNCls();
	bool init_box_model(string);
	void get_boxes(Mat img);
	void get_boxes(Mat img, vector<vector<float> > &boxes);
	void set_overlap_th(float);
	void set_confidence_th(float);
	vector<PropBox> detect(const Mat, vector<int>);
	void plotresult(const Mat &imgin, PropBox &bbox);
	void set_boxes(int nBoxes, int nDim, float* pointer);

private:
	void mat2uchar(_matrix<unsigned char>& img, Mat t);
	void image_pyramid(const Mat img);
	void project_im_rois();
	void do_forward();
	vector<PropBox> post_process(const vector<int> tgt_cls);
	bool nms(vector<Bbox> &src, vector<Bbox> &dst, float overlap);

	vector<vector<float>> boxes_;  // x1, y1, x2, y2
	vector<Blob<float>*> output_blobs;
	int box_num_;
	double scale_factor_;
	float* img_input_;
	float* roi_input_;
	vector<int> img_shape_;
	vector<int> roi_shape_;
	float overlap_th_;
	float confidence_th_;

	Model* box_model;
};



//----------------- Scene Feature Class ----------------
class SceneFeatureCls: virtual public CaffeCls {
public:
	SceneFeatureCls();
	~SceneFeatureCls();
	void set_feature_name(string);
	vector<float> extract_features(const Mat);

private:
	bool find_blobs();
	void get_input_data(const Mat, int, int);

	Blob<float>* feature_blob_;
	Blob<float>* data_blob_;
	MemoryDataLayer<float>* data_layer;
	float* img_input_;
	string feature_blob_name_;
};


#endif
