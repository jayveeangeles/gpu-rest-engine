#include "frcnn.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

#include "helper.h"
#include "common.h"
#include "gpu_allocator.h"
#include "pluginFactory.h" //
#include <chrono>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using std::string;
using GpuMat = cuda::GpuMat;
using namespace cv;

using milli = std::chrono::milliseconds;

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";

static const int SCALES = 300;
static const int MAX_SIZE = 500;
static const int NMS_MAX_OUT = 300; // This value needs to be changed as per the nmsMaxOut value set in RPROI plugin parameters in prototxt

typedef std::pair<string, box*> Prediction;

class InferenceEngine
{
public:
  InferenceEngine(const string& model_file,
                  const string& trained_file,
                  const string& trt_model,
                  const std::vector<std::string>& outputs);

  ~InferenceEngine();

  ICudaEngine* Get() const
  {
    return engine_;
  }

private:
  ICudaEngine* engine_;
  IHostMemory* serialized_model_;
};

InferenceEngine::InferenceEngine(const string& model_file,
                                 const string& trained_file,
                                 const string& trt_model,
                                 const std::vector<std::string>& outputs)
{
  FRCNNPluginFactory pluginFactorySerialize; //
  initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
  IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());

  // parse the caffe model to populate the network, then set the outputs
  INetworkDefinition* network = builder->createNetwork();

  ICaffeParser* parser = createCaffeParser();
  parser->setPluginFactoryV2(&pluginFactorySerialize); //

  auto blob_name_to_tensor = parser->parse(model_file.c_str(),
                                            trained_file.c_str(),
                                            *network,
                                            nvinfer1::DataType::kHALF);
  CHECK(blob_name_to_tensor) << "Could not parse the model";

  // specify which tensors are outputs
  for (auto& s : outputs)
    network->markOutput(*blob_name_to_tensor->find(s.c_str()));

  if (fileExists(trt_model))
  {
    gLogInfo << "Using previously generated plan file located at " << trt_model
        << std::endl;
    engine_ = loadTRTEngine(trt_model, nullptr, gLogger);
    return;
  }

  // Build the engine
  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(1 << 30);
  builder->setHalf2Mode(true);

  engine_ = builder->buildCudaEngine(*network);
  serialized_model_ = engine_->serialize();
  CHECK(engine_) << "Failed to create inference engine.";

  // write data to output file
  std::stringstream gie_model_stream;
  gie_model_stream.seekg(0, gie_model_stream.beg);
  gie_model_stream.write(static_cast<const char*>(serialized_model_->data()), serialized_model_->size());
  std::ofstream outFile;
  outFile.open(trt_model);
  outFile << gie_model_stream.rdbuf();
  outFile.close();

  gLogInfo << "Serialized plan file cached at location : " << trt_model << std::endl;

  network->destroy();
  builder->destroy();
  parser->destroy();
  pluginFactorySerialize.destroyPlugin();
  serialized_model_->destroy();
  shutdownProtobufLibrary();
}

InferenceEngine::~InferenceEngine()
{
  engine_->destroy();
}

class FRCNNDetector
{
public:
  FRCNNDetector(std::shared_ptr<InferenceEngine> engine,
            const string& label_file,
            GPUAllocator* allocator);

  ~FRCNNDetector();

  std::vector<Prediction> Detect(const Mat& img, float confthre);

private:
  void SetModel();

  void SetMean();

  void SetLabels(const string& label_file);

  void Predict(const Mat& img);

  void WrapInputLayer(std::vector<GpuMat>* input_channels);

  void Preprocess(const Mat& img,
                  std::vector<GpuMat>* input_channels);

private:
  GPUAllocator* allocator_;
  GpuMat mean_;
  std::shared_ptr<InferenceEngine> engine_;
  IExecutionContext* context_;
  std::vector<string> labels_;
  DimsCHW input_dim_;
  Size input_cv_size_;

  void* buffers_ [5];
  cudaStream_t stream_;
  int data_size_;
  int bbox_pred_size_;
  int cls_prob_size_;
  int rois_size_;

  int input_index0_;
  int input_index1_;
  int output_index0_;
  int output_index1_;
  int output_index2_;

  bool first_frame_;

  int input_width_;
  int input_height_;
  float scale_;

  float im_info_ [3];
  std::vector<float> rois_;
  std::vector<float> bbox_preds_;
  std::vector<float> cls_probs_;
  std::vector<float> pred_bboxes_;
};

FRCNNDetector::FRCNNDetector(std::shared_ptr<InferenceEngine> engine,
                       const string& label_file,
                       GPUAllocator* allocator)
    : allocator_(allocator),
      engine_(engine),
      first_frame_(true),
      input_width_(1280),
      input_height_(720)
{
  cudaError_t st = cudaStreamCreate(&stream_);
  CHECK_EQ(st, cudaSuccess) << "Could not create stream.";

  SetLabels(label_file);
  SetModel();
  SetMean();
  // Host memory for outputs
  rois_.assign(NMS_MAX_OUT * 4, 0);
  bbox_preds_.assign(NMS_MAX_OUT * (labels_.size() + 1) * 4, 0);
  cls_probs_.assign(NMS_MAX_OUT * (labels_.size() + 1) * 4, 0);

  // Predicted bounding boxes
  pred_bboxes_.assign(NMS_MAX_OUT * (labels_.size() + 1) * 4, 0);
}

FRCNNDetector::~FRCNNDetector()
{
  context_->destroy();
  cudaStreamDestroy(stream_);
  CHECK_EQ(cudaFree(buffers_[input_index0_]), cudaSuccess)  << "Could not free data layer"; ;
  CHECK_EQ(cudaFree(buffers_[input_index1_]), cudaSuccess)  << "Could not free im_info layer"; ;
  CHECK_EQ(cudaFree(buffers_[output_index0_]), cudaSuccess) << "Could not free bbox_pred layer"; ;
  CHECK_EQ(cudaFree(buffers_[output_index1_]), cudaSuccess) << "Could not free cls_prod layer"; ;
  CHECK_EQ(cudaFree(buffers_[output_index2_]), cudaSuccess) << "Could not free rois layer"; ;
}

void FRCNNDetector::SetModel()
{
  ICudaEngine* engine = engine_->Get();

  input_index0_  = engine->getBindingIndex(INPUT_BLOB_NAME0);
  input_index1_  = engine->getBindingIndex(INPUT_BLOB_NAME1);
  output_index0_ = engine->getBindingIndex(OUTPUT_BLOB_NAME0);
  output_index1_ = engine->getBindingIndex(OUTPUT_BLOB_NAME1);
  output_index2_ = engine->getBindingIndex(OUTPUT_BLOB_NAME2);

  context_ = engine->createExecutionContext();
  CHECK(context_) << "Failed to create execution context.";
  
  input_dim_ = static_cast<DimsCHW&&>(engine->getBindingDimensions(input_index0_));

  data_size_      = input_dim_.w() * input_dim_.h() * 3;
  bbox_pred_size_ = NMS_MAX_OUT * (labels_.size() + 1) * 4;
  cls_prob_size_  = NMS_MAX_OUT * (labels_.size() + 1);
  rois_size_      = NMS_MAX_OUT * 4;

  cudaError_t st = cudaMalloc(&buffers_[input_index0_], data_size_ * sizeof(float));  // data
  CHECK_EQ(st, cudaSuccess) << "Could not allocate data layer.";
  st = cudaMalloc(&buffers_[input_index1_], 3 * sizeof(float));                       // im_info
  CHECK_EQ(st, cudaSuccess) << "Could not allocate im_info layer.";
  st = cudaMalloc(&buffers_[output_index0_], bbox_pred_size_ * sizeof(float));        // bbox_pred
  CHECK_EQ(st, cudaSuccess) << "Could not allocate bbox_pred layer.";
  st = cudaMalloc(&buffers_[output_index1_], cls_prob_size_ * sizeof(float));         // cls_prob
  CHECK_EQ(st, cudaSuccess) << "Could not allocate cls_prob layer.";
  st = cudaMalloc(&buffers_[output_index2_], rois_size_ * sizeof(float));             // rois
  CHECK_EQ(st, cudaSuccess) << "Could not allocate rois layer.";

  input_cv_size_ = Size(input_dim_.w(), input_dim_.h());
}

void FRCNNDetector::SetMean()
{
    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    Scalar channel_mean = cv::Scalar(102.9801, 115.9465, 122.7717);
    Mat host_mean = Mat(input_cv_size_, CV_32FC3, channel_mean);
    mean_.upload(host_mean);
}

void FRCNNDetector::SetLabels(const string& label_file)
{
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));
}

/* Return the top N predictions. */
std::vector<Prediction> FRCNNDetector::Detect(const Mat& img, float confthre)
{
  std::vector<box *> boxes;
  std::vector<Prediction> predictions;
  Predict(img);
  float *output_data[] = {bbox_preds_.data(), cls_probs_.data(), rois_.data()};

  getBbox(boxes, output_data, 0.45, confthre, im_info_, 1, NMS_MAX_OUT, labels_.size() + 1);

  for (auto& s : boxes)
    if (s->cls > 0)
      predictions.push_back(std::make_pair(labels_[s->cls - 1], s));

  return predictions;
}

void FRCNNDetector::Predict(const Mat& img)
{
  std::vector<GpuMat> input_channels;

  WrapInputLayer(&input_channels);
  Preprocess(img, &input_channels);

  cudaError_t st = cudaMemcpyAsync(buffers_[input_index1_], im_info_, 3 * sizeof(float), cudaMemcpyHostToDevice, stream_);
  CHECK_EQ(st, cudaSuccess) << "Could not copy im_info layer.";
  context_->enqueue(1, buffers_, stream_, nullptr);
  st = cudaMemcpyAsync(bbox_preds_.data(), buffers_[output_index0_], bbox_pred_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  CHECK_EQ(st, cudaSuccess) << "Could not copy out bbox_preds layer.";
  st = cudaMemcpyAsync(cls_probs_.data(), buffers_[output_index1_], cls_prob_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  CHECK_EQ(st, cudaSuccess) << "Could not copy out cls_probs layer.";
  st = cudaMemcpyAsync(rois_.data(), buffers_[output_index2_], rois_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  CHECK_EQ(st, cudaSuccess) << "Could not copy out rois layer.";
  cudaStreamSynchronize(stream_);
}

/* Wrap the input layer of the network in separate Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void FRCNNDetector::WrapInputLayer(std::vector<GpuMat>* input_channels)
{
  int width = input_dim_.w();
  int height = input_dim_.h();
  float* input_data = reinterpret_cast<float*>(buffers_[input_index0_]);
  for (int i = 0; i < input_dim_.c(); ++i)
  {
    GpuMat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void FRCNNDetector::Preprocess(const Mat& host_img,
                            std::vector<GpuMat>* input_channels)
{
    if (first_frame_) 
  {
    input_width_ = host_img.cols;
    input_height_ = host_img.rows;

    int min_size = host_img.rows < host_img.cols ? host_img.rows : host_img.cols;
    int max_size = host_img.rows < host_img.cols ? host_img.cols : host_img.rows;

    scale_ = 1. * SCALES / min_size;

    if (max_size * scale_ > MAX_SIZE)
        scale_ = 1. * MAX_SIZE / max_size;

    first_frame_ = false;
  }

  im_info_[0] = input_height_;
  im_info_[1] = input_width_;
  im_info_[2] = scale_;

  int num_channels = input_dim_.c();
  GpuMat img(host_img, allocator_);
  /* Convert the input image to the input image format of the network. */
  GpuMat sample(allocator_);
  if (img.channels() == 3 && num_channels == 1)
    cuda::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels == 1)
    cuda::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels == 3)
    cuda::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels == 3)
    cuda::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  GpuMat sample_resized(allocator_);
  if (sample.size() != input_cv_size_)
    cuda::resize(sample, sample_resized, input_cv_size_);
  else
    sample_resized = sample;

  GpuMat sample_float(allocator_);
  if (num_channels == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  GpuMat sample_normalized(allocator_);
  cuda::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
  cuda::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == buffers_[input_index0_])
      << "Input channels are not wrapping the input layer of the network.";
}

/* By using Go as the HTTP server, we have potentially more CPU threads than
 * available GPUs and more threads can be added on the fly by the Go
 * runtime. Therefore we cannot pin the CPU threads to specific GPUs.  Instead,
 * when a CPU thread is ready for inference it will try to retrieve an
 * execution context from a queue of available GPU contexts and then do a
 * cudaSetDevice() to prepare for execution. Multiple contexts can be allocated
 * per GPU. */
class ExecContext
{
  public:
    friend ScopedContext<ExecContext>;

    static bool IsCompatible(int device)
    {
      cudaError_t st = cudaSetDevice(device);
      if (st != cudaSuccess)
        return false;

      cuda::DeviceInfo dev_info;
      if (dev_info.majorVersion() < 3)
        return false;

      return true;
    }

    ExecContext(std::shared_ptr<InferenceEngine> engine,
                const string& label_file,
                int device)
        : device_(device)
    {
      cudaError_t st = cudaSetDevice(device_);

      if (st != cudaSuccess)
        throw std::invalid_argument("could not set CUDA device");

      allocator_.reset(new GPUAllocator(1024 * 1024 * 128));
      frcnn_detector.reset(new FRCNNDetector(engine, label_file, allocator_.get()));
    }

    FRCNNDetector* TensorRTFRCNNDetector()
    {
      return frcnn_detector.get();
    }

  private:
    void Activate()
    {
      cudaError_t st = cudaSetDevice(device_);
      if (st != cudaSuccess)
        throw std::invalid_argument("could not set CUDA device");
      allocator_->reset();
    }

    void Deactivate() {}

  private:
    int device_;
    std::unique_ptr<GPUAllocator> allocator_;
    std::unique_ptr<FRCNNDetector> frcnn_detector;
};

struct frcnn_ctx
{
  ContextPool<ExecContext> pool;
};

constexpr static int kContextsPerDevice = 2;

frcnn_ctx* frcnn_initialize(char* model_file, char* trained_file, char* label_file, char* trt_model)
{
  try
  {
    ::google::InitGoogleLogging("inference_server");

    int device_count;
    cudaError_t st = cudaGetDeviceCount(&device_count);
    if (st != cudaSuccess)
      throw std::invalid_argument("could not list CUDA devices");

    ContextPool<ExecContext> pool;
    for (int dev = 0; dev < device_count; ++dev)
    {
      if (!ExecContext::IsCompatible(dev))
      {
        LOG(ERROR) << "Skipping device: " << dev;
        continue;
      }

      std::shared_ptr<InferenceEngine> engine(new InferenceEngine(model_file, trained_file, trt_model, std::vector<std::string>{OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2}));

      for (int i = 0; i < kContextsPerDevice; ++i)
      {
        std::unique_ptr<ExecContext> context(new ExecContext(engine, label_file, dev));
        pool.Push(std::move(context));
      }
    }

    if (pool.Size() == 0)
      throw std::invalid_argument("no suitable CUDA device");

    frcnn_ctx* ctx = new frcnn_ctx{std::move(pool)};
    /* Successful CUDA calls can set errno. */
    errno = 0;
    return ctx;
  }
  catch (const std::invalid_argument& ex)
  {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

const char* frcnn_detect(frcnn_ctx* ctx,
                                char* buffer, size_t length, float confthre)
{
  try
  {
    _InputArray array(buffer, length);

    Mat img = imdecode(array, -1);
    if (img.empty())
      throw std::invalid_argument("could not decode image");

    std::vector<Prediction> predictions;
    {
        /* In this scope an execution context is acquired for inference and it
          * will be automatically released back to the context pool when
          * exiting this scope. */
      ScopedContext<ExecContext> context(ctx->pool);
      auto detector = context->TensorRTFRCNNDetector();
      // auto start = std::chrono::high_resolution_clock::now();
      predictions = detector->Detect(img, confthre);
      // auto finish = std::chrono::high_resolution_clock::now();
      // std::cout << "classify() took "
      //     << std::chrono::duration_cast<milli>(finish - start).count()
      //     << " milliseconds\n";
    }

    /* Write the top N predictions in JSON format. */
    std::ostringstream os;
    os << "{\"classified\": [";
    for (size_t i = 0; i < predictions.size(); ++i)
    {
        Prediction p = predictions[i];

        os << "{\"confidence\":" << std::fixed << std::setprecision(2)
            << p.second->prob << ",";
        os << "\"label\":" << "\"" << p.first << "\"" << ",";
        os << "\"xmin\":" << std::fixed << static_cast<int>(p.second->x) << ",";
        os << "\"xmax\":" << std::fixed << static_cast<int>(p.second->y) << ",";
        os << "\"ymin\":" << std::fixed << static_cast<int>(p.second->w) << ",";
        os << "\"ymax\":" << std::fixed << static_cast<int>(p.second->h) << ",";
        os << "\"attr\": []}";
        if (i != predictions.size() - 1)
            os << ",";
    }
    os << "], \"result\": \"success\"}";

    errno = 0;
    std::string str = os.str();
    return strdup(str.c_str());
  }
  catch (const std::invalid_argument&)
  {
    errno = EINVAL;
    std::ostringstream os;
    os << "{\"classified\": [], \"result\": \"fail\"}";

    std::string str = os.str();
    return strdup(str.c_str());
  }
}

void frcnn_destroy(frcnn_ctx* ctx)
{
  delete ctx;
}
