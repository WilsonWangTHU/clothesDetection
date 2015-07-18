#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include <stdio.h>
#include <fstream>

using namespace std;

template <class numtype>
class _matrix{
public:
	numtype *value;
	int size[256];
	int dim;
	int num;
public:
	_matrix(){value = NULL;num = 0;}
	_matrix(int a){
		dim = a;
		num = 0;
		value = NULL;
	}
	_matrix(int a, int* b){
		dim = a;
		num = 1;
		for(int i=0;i<a;i++){
			size[i] = b[i];
			num *= size[i];
		}
		
		if (value== NULL){
			value = new numtype[num];
		}
		else{
			delete [] value;
			value = NULL;
			value = new numtype[num];
		}
	}
	_matrix(int a, int* b, numtype* data){
		_matrix(a, b);
		for (int i = 0;i<num;i++){
			value[i] = data[i];
		}
	}
	~_matrix(){
		if (value!=NULL){
			delete [] value;
			value = NULL;
		}
	}

	void reShape(int a, int* b){
		dim = a;
		num = 1;
		for(int i=0;i<a;i++){
			size[i] = b[i];
			num *= size[i];
		}

		if (value== NULL){
			value = new numtype[num];
		}
		else{
			delete [] value;
			value = NULL;
			value = new numtype[num];
		}
	}
	void setValue(numtype t){
		for (int i = 0; i < num; i++){
			value[i] = t;
		}
	}
	void Init(int a, int* b, numtype* data){
		this->reShape(a, b);
		for (int i = 0;i<num;i++){
			value[i] = data[i];
		}
	}

	void Init(ifstream& filein)
	{
		int a; 
		filein.read((char*)& a,sizeof(int));
        int* b = new int[a];
		filein.read((char*)b,sizeof(int)*a);

		this->reShape(a, b);
		filein.read((char*)value,sizeof(numtype)*num);
		delete []b;
	}

	_matrix<numtype>& operator=(const _matrix<numtype>& t1){
		dim = t1.dim;
		num = t1.num;
		for (int i = 0; i < dim; i++){
			size[i] = t1.size[i];
		}

		if (value == NULL){
			value = new numtype[num];
		}
		else{
			delete[] value;
			value = NULL;
			value = new numtype[num];
		}

		for (int i = 0; i < num; i++){
			value[i] = t1.value[i];
		}
		return *this;
	}

	void cat3(_matrix<numtype>* t,int count)
	{
		dim = t[0].dim;
		size[0] = t[0].size[0];
		size[1] = t[0].size[1];
		size[2] = 0;
		for (int i = 0; i < count;i++){
			size[2] += t[i].size[2];
		}
		num = 1;
		for (int i = 0; i < dim; i++){
			num *= size[i];
		}
		if (value == NULL){
			value = new numtype[num];
		}
		else{
			delete[] value;
			value = NULL;
			value = new numtype[num];
		}
		int startIndex = 0;
		for (int i = 0; i < count; i++){ 
			for (int j = 0; j < t[i].num;j++){
				value[startIndex + j] = t[i].value[j];
			}
			startIndex += t[i].num;
		}
	}
};

class _opts{
public:
	int imWidth;
	int gtWidth;
	int nPos;
	int nNeg;
	int nImgs;
	int nTrees;
	float fracFtrs;
	int minCount;
	int minChild;
	int maxDepth;
	int discretize;
	int nSamples;
	int nClasses;
	int split;
	int nOrients;
	int grdSmooth;
	int chnSmooth;
	int simSmooth;
	int normRad;
	int shrink;
	int nCells;
	int rgbd;
	int stride;
	int multiscale;
	int sharpen;
	int nTreesEval;
	int nThreads;
	int nms;
	int seed;
	int useParfor;
	int nChns;
	int nChnFtrs;
	int nSimFtrs;
	int nTotFtrs;


	float alpha;
	float beta;
	float minScore;
	int maxBoxes;
	float edgeMinMag;
	float edgeMergeThr;
	float clusterMinMag;
	float maxAspectRatio;
	float minBoxArea;
	float gamma;
	float kappa;
	
	
	void Init(ifstream& filein){
		filein.read((char*)&imWidth,sizeof(int));
		filein.read((char*)&gtWidth,sizeof(int));
		filein.read((char*)&nPos,sizeof(int));
		filein.read((char*)&nNeg,sizeof(int));
		filein.read((char*)&nImgs,sizeof(int));
		filein.read((char*)&nTrees,sizeof(int));
		filein.read((char*)&fracFtrs,sizeof(float));
		filein.read((char*)&minCount,sizeof(int));
		filein.read((char*)&minChild,sizeof(int));
		filein.read((char*)&maxDepth,sizeof(int));
		filein.read((char*)&discretize,sizeof(int));
		filein.read((char*)&nSamples,sizeof(int));
		filein.read((char*)&nClasses,sizeof(int));
		filein.read((char*)&split,sizeof(int));
		filein.read((char*)&nOrients,sizeof(int));
		filein.read((char*)&grdSmooth,sizeof(int));

		filein.read((char*)&chnSmooth,sizeof(int));
		filein.read((char*)&simSmooth,sizeof(int));
		filein.read((char*)&normRad,sizeof(int));
		filein.read((char*)&shrink,sizeof(int));
		filein.read((char*)&nCells,sizeof(int));
		filein.read((char*)&rgbd,sizeof(int));

		filein.read((char*)&stride,sizeof(int));
		filein.read((char*)&multiscale,sizeof(int));
		filein.read((char*)&sharpen,sizeof(int));
		filein.read((char*)&nTreesEval,sizeof(int));


		filein.read((char*)&nThreads,sizeof(int));
		filein.read((char*)&nms,sizeof(int));
		filein.read((char*)&seed,sizeof(int));
		filein.read((char*)&useParfor,sizeof(int));
		filein.read((char*)&nChns,sizeof(int));
		filein.read((char*)&nChnFtrs,sizeof(int));
		filein.read((char*)&nSimFtrs, sizeof(int));
		filein.read((char*)&nTotFtrs,sizeof(int));


		alpha = 0.65;
		beta = 0.75;
		minScore = 0.01;
		maxBoxes =  10000;
		edgeMinMag = 0.1;
		edgeMergeThr = 0.5;
		clusterMinMag = 0.5;
		maxAspectRatio = 3;
	    minBoxArea = 1000;
		gamma = 2;
		kappa = 1.5;
	}
};

class Model {
public:
	 _matrix<sType> thrs;
	 _matrix<int> fids;
	 _matrix<int> child;
	 _matrix<int> count;
	 _matrix<int> depth;
	 _matrix<int> nSegs;
	 _matrix<int> eBins;
	 _matrix<int> eBnds;
	 _matrix<unsigned char> segs;
	 _opts opts;
	 
	 Model() {};
	 Model(const string& n1) 
	 {
		 initmodel(n1); 
	 };
	 Model(ifstream& n1) 
	 {
		 initmodel(n1); 
	 };
	 ~Model();
	 void initmodel(const string&);
	 void initmodel(ifstream&);
};

#endif
