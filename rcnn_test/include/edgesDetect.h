#ifndef EDGESDETECT_H
#define EDGESDETECT_H

#include "config.h"
#include "model.h"
#include "chnsFunctions.h"

//void edgesDetect(_matrix<unsigned char>& img, Model& model, _matrix<float> E, _matrix<float> O);

void edgesDetect(Model& model, float* I, float* chns, float *chnsSs, int* sizeImg, _matrix<float>& E_m, _matrix<int>& ind_m);
void edgesDetect(_matrix<unsigned char>& I, Model& model, _matrix<float>& E, _matrix<float>& O);

#endif
