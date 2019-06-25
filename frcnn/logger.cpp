/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "logger.h"
#include "logging.h"

//! Decouple default reportability level for sample and TRT-API specific logging.
//! This change is required to not log TRT-API specific info by default.
//! To enable verbose logging of TRT-API use `setReportableSeverity(Severity::kINFO)`.

//! TODO: Revert gLoggerSample to gLogger to use same reportablilty level for TRT-API and samples
//! once we have support for Logger::Severity::kVERBOSE. TensorRT runtime will enable this
//! new logging level in future releases when making other ABI breaking changes.

//! gLogger is used to set default reportability level for TRT-API specific logging.
Logger gLogger{Logger::Severity::kWARNING};

//! gLoggerSample is used to set default reportability level for sample specific logging.
Logger gLoggerSample{Logger::Severity::kINFO};

#if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR > 0
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLoggerSample)};
#endif
LogStreamConsumer gLogInfo{LOG_INFO(gLoggerSample)};
LogStreamConsumer gLogWarning{LOG_WARN(gLoggerSample)};
LogStreamConsumer gLogError{LOG_ERROR(gLoggerSample)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLoggerSample)};

void setReportableSeverity(Logger::Severity severity)
{
  gLogger.setReportableSeverity(severity);
#if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR > 0
  gLogVerbose.setReportableSeverity(severity);
#endif
  gLogInfo.setReportableSeverity(severity);
  gLogWarning.setReportableSeverity(severity);
  gLogError.setReportableSeverity(severity);
  gLogFatal.setReportableSeverity(severity);
}