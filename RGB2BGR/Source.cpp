#include <iostream>
#include <vector>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>




 //Declare the CUDA kernel
//__global__ void transformRGBtoBGR(unsigned char* frameBuffer, int width, int height)
//{
//  // Calculate the row and column of the current thread
//  int row = blockIdx.y * blockDim.y + threadIdx.y;
//  int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//  // Check if the current thread is within the bounds of the frame buffer
//  if (row >= height || col >= width)
//  {
//    return;
//  }
//
//  // Calculate the index of the current pixel in the frame buffer
//  int index = row * width * 3 + col * 3;
//
//  // Swap the red and blue color values of the pixel
//  unsigned char temp = frameBuffer[index];
//  frameBuffer[index] = frameBuffer[index + 2];
//  frameBuffer[index + 2] = temp;
//}
//
//// Host function to launch the CUDA kernel
//void transformFrameBuffer(unsigned char* frameBuffer, int width, int height)
//{
//  // Calculate the number of threads and blocks needed for the kernel
//  int threadsPerBlock = 32;
//  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
//  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
//
//  // Launch the kernel
//  transformRGBtoBGR<<<numBlocksX, numBlocksY, threadsPerBlock>>>(frameBuffer, width, height);
//}

void transformToBGR_Parallel_SIMD(const std::vector<unsigned char>& frame, std::vector<unsigned char>& transformed, const size_t numPixels, const size_t numThreads)
{

    // Get the number of pixels in the frame
    //const size_t numPixels = frame.size() / 3;

    // Transform the frame using parallel threads and SIMD instructions
    //const size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; i++)
    {
        threads.emplace_back([&, i]()
        {
            // Compute the range of pixels that this thread will process
            const size_t start = i * (numPixels / numThreads);
            const size_t end = (i + 1) * (numPixels / numThreads);

            // Transform the pixels in the specified range using SIMD instructions
            for (size_t j = start; j < end; j += 8)
            {
                // Load 8 pixels from the input frame
                //const __m256i pixels = _mm256_loadu_si256((__m256i*)&frame[j * 3]);
                const __m256i pixels = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&frame[j * 3]));

                // Convert the pixels to 16-bit integers
                __m256i pixels16 = _mm256_and_si256(pixels, _mm256_set1_epi8(0xF8));

                // Move the 3 most significant bits of the G and R channels to the appropriate locations
                const __m256i g_hi = _mm256_and_si256(_mm256_srli_epi16(pixels, 5), _mm256_set1_epi8(0xE0));
                const __m256i r_lo = _mm256_and_si256(_mm256_slli_epi16(pixels, 11), _mm256_set1_epi8(0x1C));
                pixels16 = _mm256_or_si256(pixels16, _mm256_or_si256(g_hi, r_lo));

                // Store the transformed pixels in the output vector
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transformed[j * 3]), pixels16);
            }
        });
    }

    // Wait for all threads to finish
    for (std::thread& t : threads)
    {
        t.join();
    }

}


void transformToBGR_Parallel(const std::vector<unsigned char>& frame, std::vector<unsigned char>& transformed, const size_t numPixels, const size_t numThreads)
{

    // Get the number of pixels in the frame
    //const size_t numPixels = frame.size() / 3;

    // Transform the frame using parallel threads
    //const size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; i++)
    {
        threads.emplace_back([&, i]()
        {
            // Compute the range of pixels that this thread will process
            const size_t start = i * (numPixels / numThreads);
            const size_t end = (i + 1) * (numPixels / numThreads);

            // Transform the pixels in the specified range
            for (size_t j = start; j < end; j++)
            {
                // Get the R, G, and B values for the current pixel
                const unsigned char r = frame[j * 3];
                const unsigned char g = frame[j * 3 + 1];
                const unsigned char b = frame[j * 3 + 2];

                // Store the BGR pixel in the output vector
                transformed[j * 3] = b;
                transformed[j * 3 + 1] = g;
                transformed[j * 3 + 2] = r;
            }
        });
    }

    // Wait for all threads to finish
    for (std::thread& t : threads)
    {
        t.join();
    }

}


void transformToBGR_SIMD(const std::vector<unsigned char>& frame, std::vector<unsigned char>& transformed)
{

    // Get the number of pixels in the frame
    const size_t numPixels = frame.size() / 3;

    // Transform the frame using SIMD instructions
    for (size_t i = 0; i < numPixels; i += 8)
    {
        // Load 8 pixels from the input frame
        const __m256i pixels = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&frame[i * 3]));

        // Convert the pixels to 16-bit integers
        __m256i pixels16 = _mm256_and_si256(pixels, _mm256_set1_epi8(0xF8));

        // Move the 3 most significant bits of the G and R channels to the appropriate locations
        const __m256i g_hi = _mm256_and_si256(_mm256_srli_epi16(pixels, 5), _mm256_set1_epi8(0xE0));
        const __m256i r_lo = _mm256_and_si256(_mm256_slli_epi16(pixels, 11), _mm256_set1_epi8(0x1C));
        pixels16 = _mm256_or_si256(pixels16, _mm256_or_si256(g_hi, r_lo));

        // Store the transformed pixels in the output vector
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transformed[i * 3]), pixels16);
    }

}

void transformToBGR_Bitwise(const std::vector<unsigned char>& frame, std::vector<unsigned char>& transformed)
{
    
    std::transform(frame.begin(), frame.end(), transformed.begin(),
        [](unsigned char pixel) {
            return (pixel & 0xF8) | // Keep the 3 most significant bits of the B channel
                   ((pixel & 0xE0) >> 5) | // Move the 3 most significant bits of the G channel to the LSB of the B channel
                   ((pixel & 0x1C) << 11) | // Move the 3 most significant bits of the R channel to the MSB of the G channel
                   ((pixel & 0x03) << 5); // Keep the 2 least significant bits of the R channel
        });

}

int main()
{
	//const std::vector<unsigned char> frame(3840 * 2160 * 3);
	const std::vector<unsigned char> frame(1920 * 1080 * 3);
    std::vector<unsigned char> transformed(frame.size());
	const size_t numPixels = frame.size() / 3;
    const size_t numThreads = std::thread::hardware_concurrency();

	for (int i = 0; i < 5; i++)
	{
        std::cout << "Test: " << i << " ------------------------" << std::endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		transformToBGR_Bitwise(frame, transformed);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "Single_Threaded: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
		
		begin = std::chrono::steady_clock::now();
		transformToBGR_SIMD(frame, transformed);
		end = std::chrono::steady_clock::now();
		std::cout << "Single_Threaded_SIMD: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;

        begin = std::chrono::steady_clock::now();
		transformToBGR_Parallel(frame, transformed, numPixels, numThreads);
		end = std::chrono::steady_clock::now();
		std::cout << "MultiThreaded: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;

        begin = std::chrono::steady_clock::now();
		transformToBGR_Parallel_SIMD(frame, transformed, numPixels, numThreads);
		end = std::chrono::steady_clock::now();
		std::cout << "MultiThreaded_SIMD: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
        std::cout << " ------------------------" << std::endl;
	}

	getchar();
    return 0;
}