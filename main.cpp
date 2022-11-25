#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include <random>
#include <chrono>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cuda_runtime_api.h>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/optflow.hpp>

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity < Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

bool build = true;

void writeAll(const std::string &file, std::shared_ptr<nvinfer1::IHostMemory> mem)
{
    std::ofstream of(file, std::ios::binary);
    of.write((const char *)mem->data(), mem->size());
}

std::shared_ptr<ICudaEngine> readAll(const std::string &path)
{
    std::shared_ptr<ICudaEngine> result;
    std::unique_ptr<IRuntime> runtime(createInferRuntime(gLogger));
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size))
        result.reset(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    else
        std::cerr << "Can't read file " << path << std::endl;
    return result;
}

template <typename T>
size_t dim2size(Dims d)
{
    size_t size = 1;
    for (int i = 0; i < d.nbDims; i++)
    {
        size *= d.d[i];
    }
    return size * sizeof(T);
}

int main()
{
    if (build)
    {
        std::unique_ptr<IBuilder> builder(createInferBuilder(gLogger));
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
        parser->parseFromFile("flownet.onnx", int(ILogger::Severity::kWARNING));

        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        config->setFlag(BuilderFlag::kFP16);
        config->setMaxWorkspaceSize(20000000);

        std::shared_ptr<IHostMemory> rpnMem(builder->buildSerializedNetwork(*network.get(), *config.get()));
        writeAll("flownet.rtmodel", rpnMem);
    }

    auto engine = readAll("flownet.rtmodel");
    auto context = engine->createExecutionContext();

    size_t inputIndex = engine->getBindingIndex("input");
    size_t outputIndex = engine->getBindingIndex("output");
    size_t inputSize = dim2size<float>(engine->getBindingDimensions(inputIndex));
    size_t outputSize = dim2size<float>(engine->getBindingDimensions(outputIndex));

    void *buffers[2];
    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);

    cv::Mat image1 = cv::imread("data/img_050.jpg");
    cv::Mat image2 = cv::imread("data/img_051.jpg");
    cv::Mat orig1 = image1.clone();
    cv::Mat orig2 = image2.clone();

    float div_flow = 20.0;

    cv::resize(image1, image1, cv::Size(512, 512));
    cv::resize(image2, image2, cv::Size(512, 512));

    cv::cvtColor(image1, image1, cv::COLOR_BGR2RGB);
    cv::cvtColor(image2, image2, cv::COLOR_BGR2RGB);

    image1.convertTo(image1, CV_32F);
    image2.convertTo(image2, CV_32F);

    cv::divide(image1, 255.0, image1);
    cv::divide(image2, 255.0, image2);

    auto mean1 = cv::mean(image1);
    auto mean2 = cv::mean(image2);
    auto mean = (mean1 + mean2) / 2;

    cv::subtract(image1, mean, image1);
    cv::subtract(image2, mean, image2);

    std::vector<cv::Mat> channels1, channels2;
    cv::split(image1, channels1);
    cv::split(image2, channels2);

    uint8_t *ptr = (uint8_t *)buffers[inputIndex];
    cudaMemcpy(ptr, channels1[0].data, 512 * 512 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 512 * 512 * 4, channels2[0].data, 512 * 512 * 4, cudaMemcpyHostToDevice);

    cudaMemcpy(ptr + 512 * 512 * 4 * 2, channels1[1].data, 512 * 512 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 512 * 512 * 4 * 3, channels2[1].data, 512 * 512 * 4, cudaMemcpyHostToDevice);

    cudaMemcpy(ptr + 512 * 512 * 4 * 4, channels1[2].data, 512 * 512 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 512 * 512 * 4 * 5, channels2[2].data, 512 * 512 * 4, cudaMemcpyHostToDevice);

    cv::cuda::Stream stream;
    auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);

    bool success = context->enqueueV2(buffers, cudaStream, nullptr);
    stream.waitForCompletion();
    std::cout << "success: " << success << std::endl;

    ptr = (uint8_t *)buffers[outputIndex];
    cv::Mat plane1(cv::Size(128, 128), CV_32FC1);
    cv::Mat plane2(cv::Size(128, 128), CV_32FC1);

    cudaMemcpy(plane1.data, ptr, 128 * 128 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(plane2.data, ptr + 128 * 128 * 4, 128 * 128 * 4, cudaMemcpyDeviceToHost);

    cv::multiply(plane1, div_flow, plane1);
    cv::multiply(plane2, div_flow, plane2);

    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(plane1, plane2, magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
    cv::imshow("flow", bgr);
    cv::waitKey();
}
