#ifndef LOGGER_H
#define LOGGER_H

#include "logging.h"

extern Logger gLogger;
#if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR > 0
extern LogStreamConsumer gLogVerbose;
#endif
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

void setReportableSeverity(Logger::Severity severity);

#endif // LOGGER_H