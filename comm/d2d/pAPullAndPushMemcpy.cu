#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

// 错误检查宏
#define checkCudaErrors(val) \
  checkCudaErrorsImpl((val), #val, __FILE__, __LINE__)

// 错误检查实现
void checkCudaErrorsImpl(cudaError_t err, const char *func, const char *file,
                         int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error at " << file << ":" << line << " in " << func
              << " - " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

// 使用 int4 进行向量化读写的 kernel
__global__ void vectorizedMemcpyKernel(const int4 *src, int4 *dst,
                                       size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = src[idx];
  }
}

// 处理剩余元素的 kernel
__global__ void remainderMemcpyKernel(const int *src, int *dst, size_t start,
                                      size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
  if (idx < numElements) {
    dst[idx] = src[idx];
  }
}

// Push 模式 memcpy 函数
void pushModeMemcpy(int *d_src, int *d_dst, size_t numElements,
                    cudaStream_t stream) {
  const int threadsPerBlock = 256;
  size_t vectorElements = numElements / 4;
  size_t vectorBlocks =
      (vectorElements + threadsPerBlock - 1) / threadsPerBlock;

  // 使用 int4 进行主要的内存复制
  vectorizedMemcpyKernel<<<vectorBlocks, threadsPerBlock, 0, stream>>>(
      reinterpret_cast<const int4 *>(d_src), reinterpret_cast<int4 *>(d_dst),
      vectorElements);

  // 处理剩余的元素
  size_t remainderStart = vectorElements * 4;
  size_t remainderElements = numElements - remainderStart;
  if (remainderElements > 0) {
    size_t remainderBlocks =
        (remainderElements + threadsPerBlock - 1) / threadsPerBlock;
    remainderMemcpyKernel<<<remainderBlocks, threadsPerBlock, 0, stream>>>(
        d_src + remainderStart, d_dst + remainderStart, remainderStart,
        numElements);
  }
}

// Pull 模式 memcpy 函数 (与 Push 模式相同，因为 CUDA kernel 不区分 push 和
// pull)
void pullModeMemcpy(int *d_src, int *d_dst, size_t numElements,
                    cudaStream_t stream) {
  pushModeMemcpy(d_src, d_dst, numElements, stream);
}

// 使用 cudaMemcpyPeerAsync 进行设备间复制的函数
void peerMemcpy(int *d_src, int *d_dst, size_t numBytes, cudaStream_t stream) {
  cudaError_t err = cudaMemcpyPeerAsync(d_dst, 0, d_src, 1, numBytes, stream);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpyPeerAsync failed: " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {
  if (argc < 4 || (argc - 2) % 2 != 0) {
    std::cerr << "Usage: " << argv[0]
              << " <from_device1> <to_device1> [<from_device2> <to_device2> "
                 "...] --mode=<pull|push|copy_peer>"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::string mode;
  std::vector<std::pair<int, int>> devicePairs;

  // 解析命令行参数
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) {
      mode = arg.substr(7);
    } else {
      if (i + 1 < argc && argv[i + 1][0] != '-') {  // 确保是设备对
        int fromDevice = std::stoi(argv[i]);
        int toDevice = std::stoi(argv[i + 1]);
        devicePairs.emplace_back(fromDevice, toDevice);
        ++i;  // 跳过下一个参数
      } else {
        std::cerr << "Invalid device pair: " << arg << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  if (mode != "pull" && mode != "push" && mode != "copy_peer") {
    std::cerr << "Invalid mode specified. Use --mode=pull, --mode=push, or "
                 "--mode=copy_peer."
              << std::endl;
    return EXIT_FAILURE;
  }

  // 定义测试的元素数量，从 512 个元素（2KB）到约 1G 个元素（4GB）
  size_t sizes[] = {// 512, 1024, 2048, 4096, 8192, 16384, 32768,
                    // 65536, 131072, 262144, 524288, 1048576,
                    // 2097152, 4194304, 8388608, 16777216,
                    // 33554432, 67108864, 134217728, 268435456,
                    // 536870912, // 约 2GB，视设备显存而定是否启用
                    // 1073741824, // 4GB，视设备显存而定是否启用
                    4294967296};
  const int numRepeats = 1;  // 进行平均的重复次数

  // 为所有设备对启用对等访问
  for (const auto &pair : devicePairs) {
    int fromDevice = pair.first;
    int toDevice = pair.second;
    // 检查并启用对等访问
    int canAccessPeer = 0;
    checkCudaErrors(cudaSetDevice(fromDevice));
    checkCudaErrors(
        cudaDeviceCanAccessPeer(&canAccessPeer, fromDevice, toDevice));
    if (canAccessPeer) {
      cudaError_t err = cudaDeviceEnablePeerAccess(toDevice, 0);
      if (err != cudaSuccess) {
        std::cerr << "Cannot enable peer access from device " << fromDevice
                  << " to " << toDevice << " - " << cudaGetErrorString(err)
                  << std::endl;
      }
    } else {
      std::cerr << "Device " << fromDevice << " cannot access device "
                << toDevice << std::endl;
      break;
    }
    checkCudaErrors(
        cudaDeviceCanAccessPeer(&canAccessPeer, toDevice, fromDevice));
    if (canAccessPeer) {
      checkCudaErrors(cudaSetDevice(toDevice));
      cudaError_t err = cudaDeviceEnablePeerAccess(fromDevice, 0);
      if (err != cudaSuccess) {
        std::cerr << "Cannot enable peer access from device " << toDevice
                  << " to " << fromDevice << " - " << cudaGetErrorString(err)
                  << std::endl;
      }
    } else {
      std::cerr << "Device " << toDevice << " cannot access device "
                << fromDevice << std::endl;
      break;
    }
  }

  for (size_t numElements : sizes) {
    size_t sizeInBytes = numElements * sizeof(int);
    std::cout << "Testing size: " << numElements << " elements (" << sizeInBytes
              << " bytes)" << std::endl;

    std::vector<cudaStream_t> streams(devicePairs.size());
    std::vector<int *> d_srcs(devicePairs.size(), nullptr);
    std::vector<int *> d_dsts(devicePairs.size(), nullptr);
    std::vector<float> totalMilliseconds(devicePairs.size(), 0.0f);

    // 为每个设备对分配内存并创建流
    for (size_t i = 0; i < devicePairs.size(); ++i) {
      int fromDevice = devicePairs[i].first;
      int toDevice = devicePairs[i].second;

      // 分配源内存
      checkCudaErrors(cudaSetDevice(fromDevice));
      checkCudaErrors(cudaMalloc(&d_srcs[i], sizeInBytes));
      // 初始化源数据
      checkCudaErrors(cudaMemset(d_srcs[i], 1, sizeInBytes));

      // 分配目标内存
      checkCudaErrors(cudaSetDevice(toDevice));
      checkCudaErrors(cudaMalloc(&d_dsts[i], sizeInBytes));
      // 初始化目标数据
      checkCudaErrors(cudaMemset(d_dsts[i], 0, sizeInBytes));

      // 创建流
      if (mode == "pull") {
        checkCudaErrors(cudaSetDevice(toDevice));
        checkCudaErrors(
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      } else if (mode == "push") {
        checkCudaErrors(cudaSetDevice(fromDevice));
        checkCudaErrors(
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      } else if (mode == "copy_peer") {
        // 使用单一设备编号0作为源设备，1作为目标设备（请根据实际设备调整）
        checkCudaErrors(cudaSetDevice(fromDevice));  // 或 toDevice，根据需求
        checkCudaErrors(
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      }
    }

    // 进行带宽测试
    for (int repeat = 0; repeat < numRepeats; ++repeat) {
      std::vector<cudaEvent_t> startEvents(devicePairs.size());
      std::vector<cudaEvent_t> stopEvents(devicePairs.size());

      for (size_t i = 0; i < devicePairs.size(); ++i) {
        int fromDevice = devicePairs[i].first;
        int toDevice = devicePairs[i].second;

        if (mode == "pull") {
          checkCudaErrors(cudaSetDevice(toDevice));
          checkCudaErrors(cudaEventCreate(&startEvents[i]));
          checkCudaErrors(cudaEventCreate(&stopEvents[i]));

          std::cout << "i: " << i << ", startEvents: " << startEvents[i]
                    << ", stopEvents: " << stopEvents[i] << std::endl;
          checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));

          pullModeMemcpy(d_srcs[i], d_dsts[i], numElements, streams[i]);

          checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
        } else if (mode == "push") {
          checkCudaErrors(cudaSetDevice(fromDevice));

          checkCudaErrors(cudaEventCreate(&startEvents[i]));
          checkCudaErrors(cudaEventCreate(&stopEvents[i]));

          checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));

          pushModeMemcpy(d_srcs[i], d_dsts[i], numElements, streams[i]);

          checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
        } else if (mode == "copy_peer") {
          // 使用 cudaMemcpyPeerAsync 进行复制
          checkCudaErrors(cudaSetDevice(fromDevice));  // 源设备

          checkCudaErrors(cudaEventCreate(&startEvents[i]));
          checkCudaErrors(cudaEventCreate(&stopEvents[i]));

          checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));

          peerMemcpy(d_srcs[i], d_dsts[i], sizeInBytes, streams[i]);

          checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
        }
      }

      // 同步所有流并计算时间
      for (size_t i = 0; i < devicePairs.size(); ++i) {
        int fromDevice = devicePairs[i].first;
        int toDevice = devicePairs[i].second;
        float milliseconds = 0.0f;

        if (mode == "pull") {
          checkCudaErrors(cudaSetDevice(toDevice));
        } else if (mode == "push") {
          checkCudaErrors(cudaSetDevice(fromDevice));
        } else if (mode == "copy_peer") {
          checkCudaErrors(cudaSetDevice(fromDevice));  // 源设备
        }

        checkCudaErrors(cudaStreamSynchronize(streams[i]));

        checkCudaErrors(
            cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]));
        totalMilliseconds[i] += milliseconds;
        std::cout << "i: " << i << ", milliseconds: " << milliseconds
                  << std::endl;
        checkCudaErrors(cudaEventDestroy(startEvents[i]));
        checkCudaErrors(cudaEventDestroy(stopEvents[i]));
      }
    }

    // 计算并输出带宽
    for (size_t i = 0; i < devicePairs.size(); ++i) {
      int fromDevice = devicePairs[i].first;
      int toDevice = devicePairs[i].second;

      float averageMilliseconds = totalMilliseconds[i] / numRepeats;
      double bandwidth = (static_cast<double>(numElements) * sizeof(int)) /
                         (averageMilliseconds * 1e6);  // GB/s
      std::cout << "  Device Pair " << fromDevice << " -> " << toDevice
                << ": Average Bandwidth = " << bandwidth << " GB/s"
                << ", averageMilliseconds: " << averageMilliseconds
                << std::endl;

      // 释放源内存
      checkCudaErrors(cudaSetDevice(fromDevice));
      if (d_srcs[i]) {
        checkCudaErrors(cudaFree(d_srcs[i]));
        d_srcs[i] = nullptr;
      }

      // 释放目标内存
      checkCudaErrors(cudaSetDevice(toDevice));
      if (d_dsts[i]) {
        checkCudaErrors(cudaFree(d_dsts[i]));
        d_dsts[i] = nullptr;
      }

      // 销毁流
      if (mode == "pull") {
        checkCudaErrors(cudaSetDevice(toDevice));
      } else if (mode == "push") {
        checkCudaErrors(cudaSetDevice(fromDevice));
      } else if (mode == "copy_peer") {
        checkCudaErrors(cudaSetDevice(fromDevice));  // 源设备
      }
      checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    std::cout << "----------------------------------------" << std::endl;
  }

  return EXIT_SUCCESS;
}
