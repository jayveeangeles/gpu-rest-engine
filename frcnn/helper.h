#ifndef HELPER_H_
#define HELPER_H_

#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <cassert>

#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace plugin;

#include "logger.h"

// class Logger : public ILogger
// {
//   void log(Severity severity, const char* msg) override
//   {
//     // suppress info-level messages
//     if (severity != Severity::kINFO)
//       std::cout << msg << std::endl;
//   }
// };

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
bool fileExists(const std::string fileName);
nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, nvinfer1::IPluginFactory* pluginFactory,
                                     Logger& logger);

#endif
