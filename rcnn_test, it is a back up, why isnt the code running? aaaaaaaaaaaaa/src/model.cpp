//#include "model.h"
#include "../include/model.h"
void Model::initmodel(const string& modelfile)
{
	ifstream filein;
	filein.open(modelfile.c_str(), ios::binary);
	initmodel(filein);
	filein.close();
}

void Model::initmodel(ifstream& filein)
{
	opts.Init(filein);
	thrs.Init(filein);
	fids.Init(filein);
	child.Init(filein);
	count.Init(filein);
	depth.Init(filein);
	nSegs.Init(filein);
	eBins.Init(filein);
	eBnds.Init(filein);
	segs.Init(filein);
}

Model::~Model()
{
}
