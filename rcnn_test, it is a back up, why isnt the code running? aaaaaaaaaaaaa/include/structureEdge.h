#ifndef STRUCTUREEDGE_H
#define STRUCTUREEDGE_H

#include "config.h"
#include "model.h"
#include "edgeBoxes.h"
#include "edgesDetect.h"
#include "edgesNms.h"
#include "chnsFunctions.h"

void structureEdge(_matrix<unsigned char>& img, Model& model, vector<bb>& bbs);

#endif
