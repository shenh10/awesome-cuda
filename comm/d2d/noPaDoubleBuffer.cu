#include <cuda_runtime.h>

#include <chrono>  // 用于高精度计时器
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// 错误检查宏
#define checkCudaErrors(val) \
  checkCudaErrorsImpl((val), #val, __FILE__, __LINE__)

void checkCudaErrorsImpl(cudaError_t err, const char *func, const char *file,
                         int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error at " << file << ":" << line << " in " << func
              << " - " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}
// 资源管理类
template <typename T>
class PinnedMemoryPool {
 public:
  PinnedMemoryPool(size_t size, int srcDevice, int dstDevice)
      : size_(size), srcDevice_(srcDevice), dstDevice_(dstDevice) {
    checkCudaErrors(cudaMallocHost(&buffer_[0], size * sizeof(T)));
    checkCudaErrors(cudaMallocHost(&buffer_[1], size * sizeof(T)));

    for (int i = 0; i < 2; ++i) {
      // 在源设备上创建源事件
      checkCudaErrors(cudaSetDevice(srcDevice_));
      checkCudaErrors(
          cudaEventCreateWithFlags(&srcEvents_[i], cudaEventDisableTiming));

      // 在目标设备上创建目标事件
      checkCudaErrors(cudaSetDevice(dstDevice_));
      checkCudaErrors(
          cudaEventCreateWithFlags(&dstEvents_[i], cudaEventDisableTiming));
    }
  }

  ~PinnedMemoryPool() {
    // 在对应的设备上销毁事件
    cudaSetDevice(srcDevice_);
    for (int i = 0; i < 2; ++i) {
      cudaEventDestroy(srcEvents_[i]);
    }

    cudaSetDevice(dstDevice_);
    for (int i = 0; i < 2; ++i) {
      cudaEventDestroy(dstEvents_[i]);
    }

    // 释放固定内存（可以在任意设备上下文中进行）
    for (int i = 0; i < 2; ++i) {
      cudaFreeHost(buffer_[i]);
    }
  }

  T *getBuffer(int index) { return buffer_[index]; }
  cudaEvent_t getSrcEvent(int index) { return srcEvents_[index]; }
  cudaEvent_t getDstEvent(int index) { return dstEvents_[index]; }
  size_t getSize() const { return size_; }

 private:
  T *buffer_[2] = {nullptr, nullptr};
  cudaEvent_t srcEvents_[2];
  cudaEvent_t dstEvents_[2];
  size_t size_;
  int srcDevice_;
  int dstDevice_;
};

// 修改后的传输函数
template <typename T>
void nonPeerD2DCopyWithDoublePinned(const T *d_src, int srcDevice, T *d_dst,
                                    int dstDevice, size_t numElements,
                                    cudaStream_t srcStream,
                                    cudaStream_t dstStream,
                                    PinnedMemoryPool<T> &memPool) {
  const size_t CHUNK_SIZE = memPool.getSize();
  const size_t numChunks = (numElements + CHUNK_SIZE - 1) / CHUNK_SIZE;

  int currentBuffer = 0;
  size_t offset = 0;

  // 启动第一次传输
  if (numChunks > 0) {
    size_t currentChunkSize = std::min(CHUNK_SIZE, numElements);
    checkCudaErrors(cudaSetDevice(srcDevice));
    checkCudaErrors(cudaMemcpyAsync(memPool.getBuffer(currentBuffer), d_src,
                                    currentChunkSize * sizeof(T),
                                    cudaMemcpyDeviceToHost, srcStream));
    checkCudaErrors(
        cudaEventRecord(memPool.getSrcEvent(currentBuffer), srcStream));
  }

  // 处理所有完整的块
  for (size_t chunk = 1; chunk < numChunks; ++chunk) {
    int nextBuffer = 1 - currentBuffer;
    size_t nextOffset = chunk * CHUNK_SIZE;
    size_t currentChunkSize = std::min(CHUNK_SIZE, numElements - offset);
    size_t nextChunkSize = std::min(CHUNK_SIZE, numElements - nextOffset);

    checkCudaErrors(cudaSetDevice(dstDevice));
    checkCudaErrors(
        cudaStreamWaitEvent(dstStream, memPool.getSrcEvent(currentBuffer)));
    checkCudaErrors(cudaMemcpyAsync(
        d_dst + offset, memPool.getBuffer(currentBuffer),
        currentChunkSize * sizeof(T), cudaMemcpyHostToDevice, dstStream));
    checkCudaErrors(
        cudaEventRecord(memPool.getDstEvent(currentBuffer), dstStream));

    checkCudaErrors(cudaSetDevice(srcDevice));
    checkCudaErrors(
        cudaStreamWaitEvent(srcStream, memPool.getDstEvent(nextBuffer)));
    checkCudaErrors(cudaMemcpyAsync(
        memPool.getBuffer(nextBuffer), d_src + nextOffset,
        nextChunkSize * sizeof(T), cudaMemcpyDeviceToHost, srcStream));
    checkCudaErrors(
        cudaEventRecord(memPool.getSrcEvent(nextBuffer), srcStream));

    offset = nextOffset;
    currentBuffer = nextBuffer;
  }

  // 处理最后一块数据
  if (numChunks > 0) {
    size_t currentChunkSize = std::min(CHUNK_SIZE, numElements - offset);
    checkCudaErrors(cudaSetDevice(dstDevice));
    checkCudaErrors(
        cudaStreamWaitEvent(dstStream, memPool.getSrcEvent(currentBuffer)));
    checkCudaErrors(cudaMemcpyAsync(
        d_dst + offset, memPool.getBuffer(currentBuffer),
        currentChunkSize * sizeof(T), cudaMemcpyHostToDevice, dstStream));
    checkCudaErrors(
        cudaEventRecord(memPool.getDstEvent(currentBuffer), dstStream));
  }
}

template <typename T>
void nonPeerD2DCopyProgressive(const T *d_src, int srcDevice, T *d_dst,
                               int dstDevice, size_t numElements,
                               cudaStream_t srcStream, cudaStream_t dstStream,
                               PinnedMemoryPool<T> &memPool) {
  const size_t INITIAL_CHUNK_SIZE = 512 * 1024 / sizeof(T);
  const size_t MAX_CHUNK_SIZE = memPool.getSize();
  const float GROWTH_FACTOR = 1.2f;

  size_t remainingElements = numElements;
  size_t srcOffset = 0;
  size_t dstOffset = 0;
  int currentBuffer = 0;
  size_t currentChunkSize = INITIAL_CHUNK_SIZE;

  // 启动第一次传输
  currentChunkSize = std::min(currentChunkSize, remainingElements);
  checkCudaErrors(cudaSetDevice(srcDevice));
  checkCudaErrors(cudaMemcpyAsync(memPool.getBuffer(0), d_src,
                                  currentChunkSize * sizeof(T),
                                  cudaMemcpyDeviceToHost, srcStream));
  checkCudaErrors(cudaEventRecord(memPool.getSrcEvent(0), srcStream));

  remainingElements -= currentChunkSize;
  srcOffset += currentChunkSize;

  while (remainingElements > 0) {
    int nextBuffer = 1 - currentBuffer;
    size_t nextChunkSize =
        static_cast<size_t>(currentChunkSize * GROWTH_FACTOR);
    nextChunkSize = std::min(nextChunkSize, MAX_CHUNK_SIZE);
    nextChunkSize = std::min(nextChunkSize, remainingElements);

    checkCudaErrors(cudaSetDevice(dstDevice));
    checkCudaErrors(
        cudaStreamWaitEvent(dstStream, memPool.getSrcEvent(currentBuffer)));
    checkCudaErrors(cudaMemcpyAsync(
        d_dst + dstOffset, memPool.getBuffer(currentBuffer),
        currentChunkSize * sizeof(T), cudaMemcpyHostToDevice, dstStream));
    checkCudaErrors(
        cudaEventRecord(memPool.getDstEvent(currentBuffer), dstStream));

    checkCudaErrors(cudaSetDevice(srcDevice));
    if (remainingElements > nextChunkSize) {
      checkCudaErrors(
          cudaStreamWaitEvent(srcStream, memPool.getDstEvent(nextBuffer)));
    }

    checkCudaErrors(cudaMemcpyAsync(
        memPool.getBuffer(nextBuffer), d_src + srcOffset,
        nextChunkSize * sizeof(T), cudaMemcpyDeviceToHost, srcStream));
    checkCudaErrors(
        cudaEventRecord(memPool.getSrcEvent(nextBuffer), srcStream));

    dstOffset += currentChunkSize;
    srcOffset += nextChunkSize;
    remainingElements -= nextChunkSize;
    currentChunkSize = nextChunkSize;
    currentBuffer = nextBuffer;
  }

  // 处理最后一块
  checkCudaErrors(cudaSetDevice(dstDevice));
  checkCudaErrors(
      cudaStreamWaitEvent(dstStream, memPool.getSrcEvent(currentBuffer)));
  checkCudaErrors(cudaMemcpyAsync(
      d_dst + dstOffset, memPool.getBuffer(currentBuffer),
      currentChunkSize * sizeof(T), cudaMemcpyHostToDevice, dstStream));
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <from_device1> <to_device1> [<from_device2> <to_device2> "
                 "...] [--method=<double|progressive>]"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::pair<int, int>> devicePairs;
  bool useProgressive = true;  // 默认使用progressive方法

  // 解析命令行参数
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--method=") == 0) {
      std::string method = arg.substr(9);
      useProgressive = (method != "double");
    } else if (i + 1 < argc) {
      int fromDevice = std::stoi(argv[i]);
      int toDevice = std::stoi(argv[i + 1]);
      devicePairs.emplace_back(fromDevice, toDevice);
      ++i;
    }
  }

  // 测试大小数组
  size_t sizes[] = {
      //   67108864   // 64MB
      1073741824,  // 1GB
  };

  const int numRepeats = 5;
  const size_t MAX_BUFFER_SIZE = 64 * 1024 * 1024;  // 64MB

  for (size_t numElements : sizes) {
    std::cout << "\nTesting size: " << numElements << " elements ("
              << std::fixed << std::setprecision(2)
              << (numElements * sizeof(int) / (1024.0 * 1024.0)) << " MB)"
              << std::endl;
    // 准备设备内存和流
    std::vector<int *> d_srcs;
    std::vector<int *> d_dsts;
    std::vector<cudaStream_t> srcStreams;
    std::vector<cudaStream_t> dstStreams;

    // 为每个设备对分配资源
    for (const auto &pair : devicePairs) {
      int fromDevice = pair.first;
      int toDevice = pair.second;
      size_t sizeInBytes =
          numElements *
          sizeof(int);  // 修改这里使用实际需要的大小，而不是MAX_BUFFER_SIZE

      // 在源设备上分配内存和创建流
      checkCudaErrors(cudaSetDevice(fromDevice));
      int *d_src;
      checkCudaErrors(cudaMalloc(&d_src, sizeInBytes));
      d_srcs.push_back(d_src);

      cudaStream_t srcStream;
      checkCudaErrors(cudaStreamCreate(&srcStream));
      srcStreams.push_back(srcStream);

      // 在目标设备上分配内存和创建流
      checkCudaErrors(cudaSetDevice(toDevice));
      int *d_dst;
      checkCudaErrors(cudaMalloc(&d_dst, sizeInBytes));
      d_dsts.push_back(d_dst);

      cudaStream_t dstStream;
      checkCudaErrors(cudaStreamCreate(&dstStream));
      dstStreams.push_back(dstStream);

      // 初始化源数据
      checkCudaErrors(cudaSetDevice(fromDevice));
      checkCudaErrors(cudaMemset(d_src, 0xAB, sizeInBytes));
    }

    // 为每个设备对创建资源
    std::vector<std::unique_ptr<PinnedMemoryPool<int>>> memPools;
    std::vector<cudaEvent_t> startEvents, stopEvents;
    std::vector<float> totalMilliseconds(devicePairs.size(), 0.0f);

    // 初始化资源
    for (const auto &pair : devicePairs) {
      memPools.push_back(
          std::unique_ptr<PinnedMemoryPool<int>>(new PinnedMemoryPool<int>(
              MAX_BUFFER_SIZE / sizeof(int), pair.first, pair.second)));

      cudaEvent_t startEvent, stopEvent;
      checkCudaErrors(cudaSetDevice(pair.second));
      checkCudaErrors(cudaEventCreate(&startEvent));
      checkCudaErrors(cudaEventCreate(&stopEvent));

      startEvents.push_back(startEvent);
      stopEvents.push_back(stopEvent);
    }

    // 执行多次重复测试
    for (int repeat = 0; repeat < numRepeats; ++repeat) {
      std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>
          startTimes(devicePairs.size());
      // 启动所有传输
      for (size_t i = 0; i < devicePairs.size(); ++i) {
        int fromDevice = devicePairs[i].first;
        int toDevice = devicePairs[i].second;

        // 记录每个传输的开始时间
        startTimes[i] = std::chrono::high_resolution_clock::now();

        // 记录开始时间
        checkCudaErrors(cudaSetDevice(toDevice));
        checkCudaErrors(cudaEventRecord(startEvents[i], dstStreams[i]));

        // 执行传输
        if (useProgressive) {
          nonPeerD2DCopyProgressive(d_srcs[i], fromDevice, d_dsts[i], toDevice,
                                    numElements, srcStreams[i], dstStreams[i],
                                    *memPools[i]);
        } else {
          nonPeerD2DCopyWithDoublePinned(d_srcs[i], fromDevice, d_dsts[i],
                                         toDevice, numElements, srcStreams[i],
                                         dstStreams[i], *memPools[i]);
        }

        // 记录结束时间
        checkCudaErrors(cudaSetDevice(toDevice));
        checkCudaErrors(cudaEventRecord(stopEvents[i], dstStreams[i]));
      }

      // 等待所有传输完成并收集时间
      for (size_t i = 0; i < devicePairs.size(); ++i) {
        // 在目标设备上下文中同步结束事件
        checkCudaErrors(cudaSetDevice(devicePairs[i].second));
        checkCudaErrors(cudaEventSynchronize(stopEvents[i]));

        float milliseconds = 0.0f;
        checkCudaErrors(
            cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]));
        // 记录CPU结束时间并计算该传输对的耗时，这个比GPU上时间更准确
        auto endTime = std::chrono::high_resolution_clock::now();
        milliseconds =
            std::chrono::duration<float, std::milli>(endTime - startTimes[i])
                .count();

        totalMilliseconds[i] += milliseconds;
      }

      // 可选：在所有设备上同步以确保完全完成
      for (size_t i = 0; i < devicePairs.size(); ++i) {
        checkCudaErrors(cudaSetDevice(devicePairs[i].first));
        checkCudaErrors(cudaStreamSynchronize(srcStreams[i]));
        checkCudaErrors(cudaSetDevice(devicePairs[i].second));
        checkCudaErrors(cudaStreamSynchronize(dstStreams[i]));
      }
    }

    // 输出结果
    std::cout << std::setw(15) << "From GPU" << std::setw(15) << "To GPU"
              << std::setw(15) << "Bandwidth" << std::setw(15) << "Avg Time"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (size_t i = 0; i < devicePairs.size(); ++i) {
      float averageMilliseconds = totalMilliseconds[i] / numRepeats;
      double bandwidth = (static_cast<double>(numElements * sizeof(int))) /
                         (averageMilliseconds * 1e6);  // GB/s

      std::cout << std::setw(15) << devicePairs[i].first << std::setw(15)
                << devicePairs[i].second << std::setw(15) << std::fixed
                << std::setprecision(2) << bandwidth << std::setw(15)
                << averageMilliseconds << std::endl;
    }

    // 清理本轮测试的所有资源
    for (size_t i = 0; i < devicePairs.size(); ++i) {
      int fromDevice = devicePairs[i].first;
      int toDevice = devicePairs[i].second;

      // 清理事件
      checkCudaErrors(cudaSetDevice(fromDevice));
      checkCudaErrors(cudaEventDestroy(startEvents[i]));
      checkCudaErrors(cudaSetDevice(toDevice));
      checkCudaErrors(cudaEventDestroy(stopEvents[i]));

      // 清理内存和流
      checkCudaErrors(cudaSetDevice(fromDevice));
      checkCudaErrors(cudaFree(d_srcs[i]));
      checkCudaErrors(cudaStreamDestroy(srcStreams[i]));

      checkCudaErrors(cudaSetDevice(toDevice));
      checkCudaErrors(cudaFree(d_dsts[i]));
      checkCudaErrors(cudaStreamDestroy(dstStreams[i]));
    }
    std::cout << std::string(60, '-') << std::endl;
  }

  return EXIT_SUCCESS;
}
