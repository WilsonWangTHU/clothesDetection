#include <stdio.h>
#include "CaffeCls.hpp"
#include "imagehelper.hpp"
#include "tinyxml2.hpp"
#include <string.h>
#include <stdio.h>

using namespace std;
using namespace sdktest;
using namespace tinyxml2;
const int INVALID_IMAGE = -1;
class clothReader {
	public:
		static bool readClothClass(string labelName, int* classtype, float* position);
};
