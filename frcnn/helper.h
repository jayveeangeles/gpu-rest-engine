#ifndef HELPER_H_
#define HELPER_H_

#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>

struct box {
  float x;
  float y;
  float w;
  float h;
  int cls;
  float prob;
  box(float sx, float sy, float sw, float sh, int scls, float sprob) {
      x = sx; y = sy; w = sw; h = sh; cls = scls; prob = sprob;
  }
};

void getBbox(std::vector<box*>& boxes, float **outputData, float nmsThresh, float detectThresh, float* imInfo, int batchSize, const int nmsMaxOut, const int outputClsSize);

#endif
