#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

// Enhanced error checking macro
#define checkCudaErrors(val) \
  checkCudaErrorsImpl((val), #val, __FILE__, __LINE__)

void checkCudaErrorsImpl(cudaError_t err, const char* func, const char* file,
                         int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error at " << file << ":" << line << " in " << func
              << " - " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <from_device1> <to_device1> [<from_device2> <to_device2> "
                 "...] [--enable-peer-access]"
              << std::endl;
    return EXIT_FAILURE;
  }

  bool enablePeerAccess = false;
  std::vector<std::pair<int, int>> devicePairs;

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--enable-peer-access") {
      enablePeerAccess = true;
    } else {
      if (i + 1 < argc) {
        int fromDevice = std::stoi(argv[i]);
        int toDevice = std::stoi(argv[i + 1]);
        devicePairs.emplace_back(fromDevice, toDevice);
        ++i;  // Skip the next argument as it's part of the device pair
      }
    }
  }

  // Enable peer access between devices if possible and requested
  if (enablePeerAccess) {
    for (const auto& pair : devicePairs) {
      int fromDevice = pair.first;
      int toDevice = pair.second;

      int canAccessPeer = 0;
      checkCudaErrors(
          cudaDeviceCanAccessPeer(&canAccessPeer, fromDevice, toDevice));
      if (canAccessPeer) {
        cudaSetDevice(fromDevice);
        cudaError_t err = cudaDeviceEnablePeerAccess(toDevice, 0);
        if (err == cudaSuccess) {
          std::cout << "Peer access enabled from device " << fromDevice
                    << " to device " << toDevice << std::endl;
        } else {
          std::cout << "Failed to enable peer access from device " << fromDevice
                    << " to device " << toDevice << ": "
                    << cudaGetErrorString(err) << std::endl;
        }
      } else {
        std::cout << "Peer access not supported from device " << fromDevice
                  << " to device " << toDevice << std::endl;
      }

      checkCudaErrors(
          cudaDeviceCanAccessPeer(&canAccessPeer, toDevice, fromDevice));
      if (canAccessPeer) {
        cudaSetDevice(toDevice);
        cudaError_t err = cudaDeviceEnablePeerAccess(fromDevice, 0);
        if (err == cudaSuccess) {
          std::cout << "Peer access enabled from device " << toDevice
                    << " to device " << fromDevice << std::endl;
        } else {
          std::cout << "Failed to enable peer access from device " << toDevice
                    << " to device " << fromDevice << ": "
                    << cudaGetErrorString(err) << std::endl;
        }
      } else {
        std::cout << "Peer access not supported from device " << toDevice
                  << " to device " << fromDevice << std::endl;
      }
    }
  }

  size_t sizes[] = {
      // 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
      // 1048576, 2097152, 4194304, 8388608, 16777216, 33554432,
      // 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
      4294967296};           // 2KB - 4GB
  const int numRepeats = 1;  // Number of repetitions for averaging

  for (size_t size : sizes) {
    std::cout << "Testing size: " << size << " bytes" << std::endl;

    std::vector<cudaStream_t> streams(devicePairs.size());
    std::vector<void*> d_srcs(devicePairs.size());
    std::vector<void*> d_dsts(devicePairs.size());
    std::vector<float> totalMilliseconds(devicePairs.size(), 0.0f);

    for (size_t i = 0; i < devicePairs.size(); ++i) {
      int fromDevice = devicePairs[i].first;
      int toDevice = devicePairs[i].second;

      // Allocate source memory on fromDevice
      cudaSetDevice(fromDevice);
      checkCudaErrors(cudaMalloc(&d_srcs[i], size));

      // Allocate destination memory on toDevice
      cudaSetDevice(toDevice);
      checkCudaErrors(cudaMalloc(&d_dsts[i], size));

      // Create stream on fromDevice
      cudaSetDevice(fromDevice);  // Set to fromDevice before creating stream
      checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    for (int repeat = 0; repeat < numRepeats; ++repeat) {
      std::vector<cudaEvent_t> startEvents(devicePairs.size());
      std::vector<cudaEvent_t> stopEvents(devicePairs.size());

      for (size_t i = 0; i < devicePairs.size(); ++i) {
        int fromDevice = devicePairs[i].first;
        // Ensure device is set to fromDevice
        cudaSetDevice(
            fromDevice);  // Set device to fromDevice where stream[i] resides

        checkCudaErrors(cudaEventCreate(&startEvents[i]));
        checkCudaErrors(cudaEventCreate(&stopEvents[i]));

        checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));
        checkCudaErrors(cudaMemcpyAsync(d_dsts[i], d_srcs[i], size,
                                        cudaMemcpyDeviceToDevice, streams[i]));
        checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
      }

      // Synchronize all streams to ensure all operations for the current size
      // are completed
      for (size_t i = 0; i < devicePairs.size(); ++i) {
        int fromDevice = devicePairs[i].first;
        cudaSetDevice(fromDevice);  // Ensure device is set before synchronizing

        checkCudaErrors(cudaStreamSynchronize(streams[i]));

        float milliseconds = 0;
        checkCudaErrors(
            cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]));
        totalMilliseconds[i] += milliseconds;

        checkCudaErrors(cudaEventDestroy(startEvents[i]));
        checkCudaErrors(cudaEventDestroy(stopEvents[i]));
      }
    }

    for (size_t i = 0; i < devicePairs.size(); ++i) {
      int fromDevice = devicePairs[i].first;
      int toDevice = devicePairs[i].second;

      float averageMilliseconds = totalMilliseconds[i] / numRepeats;
      double bandwidth = (size * 1e-9) / (averageMilliseconds * 1e-3);
      std::cout << "  Device Pair " << fromDevice << " -> " << toDevice
                << ": Average Bandwidth = " << bandwidth
                << " GB/s, milliseconds " << averageMilliseconds << std::endl;

      // Free source memory on fromDevice
      cudaSetDevice(fromDevice);
      checkCudaErrors(cudaFree(d_srcs[i]));

      // Free destination memory on toDevice
      cudaSetDevice(toDevice);
      checkCudaErrors(cudaFree(d_dsts[i]));

      // Destroy stream on fromDevice
      cudaSetDevice(
          fromDevice);  // Ensure device is set before destroying stream
      checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
  }

  return EXIT_SUCCESS;
}
