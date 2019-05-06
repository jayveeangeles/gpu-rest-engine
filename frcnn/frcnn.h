#ifndef FRCNN_H_
#define FRCNN_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct frcnn_ctx frcnn_ctx;

frcnn_ctx* frcnn_initialize(char* model_file, char* trained_file, char* label_file, char* trt_model);

const char* frcnn_detect(frcnn_ctx* ctx,
                                char* buffer, size_t length);

void frcnn_destroy(frcnn_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif
