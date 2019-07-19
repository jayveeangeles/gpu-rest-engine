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

#if NV_TENSORRT_MAJOR >= 5
#include "logger.h"
#else
// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:

    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

static Logger gLogger;
#endif

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

void getBoxes(std::vector<box*>& boxes, float **outputData, float nmsThresh, float detectThresh, float* imInfo, int batchSize, const int nmsMaxOut, const int outputClsSize);
bool fileExists(const std::string fileName);
nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, nvinfer1::IPluginFactory* pluginFactory,
                                     Logger& logger);

#endif
