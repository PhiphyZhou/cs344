// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"

// blur function not using shared memory 
__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.
  
  // Calculate and check the absolute image position
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if ( c >= numCols || r >= numRows ){    
	return;
  }
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
  
  float result = 0.f;
  //For every value in the filter around the pixel (c, r)
  for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
	for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
	  //Find the global image position for this filter position
	  //clamp to boundary of the image
	  int image_r = min(max(r + filter_r, 0), static_cast<int>(numRows - 1));
	  int image_c = min(max(c + filter_c, 0), static_cast<int>(numCols - 1));

	  float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
	  float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

	  result += image_value * filter_value;
	}
  }

  outputChannel[r * numCols + c] = static_cast<unsigned char>(result);
}

// blur function using shared memory 
__global__
void gaussian_blur_sh(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.
  
  // Calculate and check the absolute image position
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if ( c >= numCols || r >= numRows ){    
	return;
  }
  
  // use shared memory for both filter and image channel
  extern __shared__ float sh_filter[];
  unsigned char* sh_image = reinterpret_cast<unsigned char*>(&sh_filter[filterWidth*filterWidth]);
  
  // fill in the shared memory
  int global_idx = r * numCols + c; // index in the whole image array
  int offset = filterWidth/2; 
  int shared_idx = (threadIdx.y+offset) * (blockDim.x+offset*2) + threadIdx.x + offset; // index in the image block
  sh_image[shared_idx] = inputChannel[global_idx]; // copy corresponding pixel into shared memory 
  
  // extra work to load the offset of the image block and the filter
  if (threadIdx.x < offset){
  	// clamp if global index exceeds the left boundary
  	int cc = max(c-offset,0);
  	sh_image[shared_idx-offset] = inputChannel[r*numCols+cc];
  }
  else if (threadIdx.x > blockDim.x-offset-1 ||  // deal with the rightmost non-full blocks
  	(blockIdx.x == gridDim.x - 1 && numCols%blockDim.x != 0 && 
  	 threadIdx.x > numCols % blockDim.x-offset-1)) {
    // clamp if global index exceeds the right boundary
    int cc = min(c+offset, numCols-1);
  	sh_image[shared_idx+offset] = inputChannel[r*numCols+cc];
  }

     
  if (threadIdx.y < offset){
    int rr = max(r-offset,0);
    int sidx = threadIdx.y*(blockDim.x+offset*2) + threadIdx.x + offset; 
    sh_image[sidx] = inputChannel[rr*numCols+c];
    // take care of the pixels in the corners
    if (threadIdx.x < offset){
		int cc = max(c-offset,0);
		sh_image[sidx-offset] = inputChannel[rr*numCols+cc];
  	}
    else if (threadIdx.x > blockDim.x-offset-1 ||
    (blockIdx.x == gridDim.x - 1 && numCols%blockDim.x != 0 && 
  	 threadIdx.x > numCols % blockDim.x-offset-1)) {
		int cc = min(c+offset, numCols-1);
		sh_image[sidx+offset] = inputChannel[rr*numCols+cc];
    }
  }
  else if (threadIdx.y > blockDim.y -offset -1 || // deal with the bottom non-full block
    (blockIdx.y == gridDim.y - 1 && numRows%blockDim.y != 0 && 
  	 threadIdx.y > numRows % blockDim.y-offset-1)){
    int rr = min(r+offset, numRows-1);
    int sidx = (threadIdx.y+2*offset)*(blockDim.x+offset*2) + threadIdx.x + offset; 
    sh_image[sidx] = inputChannel[rr*numCols+c];
    if (threadIdx.x < offset){
		int cc = max(c-offset,0);
		sh_image[sidx-offset] = inputChannel[rr*numCols+cc];
  	}
    else if (threadIdx.x > blockDim.x-offset-1 ||
     (blockIdx.x == gridDim.x - 1 && numCols%blockDim.x != 0 && 
  	 threadIdx.x > numCols % blockDim.x-offset-1)) {
		int cc = min(c+offset, numCols-1);
		sh_image[sidx+offset] = inputChannel[rr*numCols+cc];
    }
  }

 //  __syncthreads(); 
 //  outputChannel[r*numCols+c] = sh_image[shared_idx];

  // load the filter to shared memory
  if (threadIdx.x < filterWidth && threadIdx.y < filterWidth){
    int fidx = threadIdx.y * filterWidth + threadIdx.x;
  	sh_filter[fidx] = filter[fidx];
  }
  // outputChannel[r*numCols+c] = sh_image[shared_idx];

  __syncthreads(); 

//outputChannel[r*numCols+c] = sh_image[shared_idx];
  // do convolution using the shared memory
  float result = 0.f;
  //For every value in the filter around the pixel (c, r)
  for (int filter_r = -offset; filter_r <= offset; ++filter_r) {
	for (int filter_c = -offset; filter_c <= offset; ++filter_c) {
      // extended image block already built, no need to clamp		
	  int image_r = threadIdx.y + offset + filter_r;
	  int image_c = threadIdx.x + offset + filter_c;

	  float image_value = static_cast<float>(sh_image[image_r * (blockDim.x+2*offset) + image_c]);
	  float filter_value = sh_filter[(filter_r + offset) * filterWidth + filter_c + offset];

	  result += image_value * filter_value;
	}
  }

  outputChannel[r * numCols + c] = static_cast<unsigned char>(result); // write to global memory
  
//  outputChannel[r*numCols+c] = sh_image[shared_idx];
}


//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // Calculate the absolute image position
  size_t c = blockIdx.x * blockDim.x + threadIdx.x;
  size_t r = blockIdx.y * blockDim.y + threadIdx.y;
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  if ( c >= numCols || r >= numRows ){    
	return;
  }
  
 // split channels
  size_t index = r * numCols + c;
  redChannel[index] = inputImageRGBA[index].x;
  greenChannel[index] = inputImageRGBA[index].y;
  blueChannel[index] = inputImageRGBA[index].z;
  
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  size_t fsz = sizeof(float) * filterWidth * filterWidth;
  checkCudaErrors(cudaMalloc(&d_filter, fsz));

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, fsz, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const size_t bsz = 32;
  const dim3 blockSize(bsz,bsz,1);

  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize((numCols+bsz-1)/bsz, (numRows+bsz-1)/bsz, 1);

  //TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize,blockSize>>>(d_inputImageRGBA, numRows, numCols, 
  											d_red, d_green, d_blue);
  				
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  std::cout<<"separated Channels!"<<std::endl;

  // choose to use shared memory or not
  bool share = true;
  
  if (share == false){
	  //TODO: Call your convolution kernel here 3 times, once for each color channel.
	  gaussian_blur<<<gridSize,blockSize>>>(d_red, d_redBlurred, numRows, numCols,
											d_filter, filterWidth);
	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());                 						
	  gaussian_blur<<<gridSize,blockSize>>>(d_green, d_greenBlurred, numRows, numCols,
											d_filter, filterWidth);
	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	  gaussian_blur<<<gridSize,blockSize>>>(d_blue, d_blueBlurred, numRows, numCols,
											d_filter, filterWidth);
	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }
  else {
  	// compute shared memory size
	size_t sh_fsz = filterWidth*filterWidth; // filter: number of floats
	size_t sh_isz = (bsz+filterWidth-1)*(bsz+filterWidth-1); // image block: number of uchars
	size_t sh_sz = sh_fsz*sizeof(float)+sh_isz*sizeof(unsigned char); // total shared size in bytes
	
	gaussian_blur_sh<<<gridSize,blockSize,sh_sz>>>(d_red, d_redBlurred, 
											numRows, numCols, d_filter, filterWidth);
	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());                 						
	  gaussian_blur_sh<<<gridSize,blockSize,sh_sz>>>(d_green, d_greenBlurred, 
											numRows, numCols, d_filter, filterWidth);
	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	  gaussian_blur_sh<<<gridSize,blockSize,sh_sz>>>(d_blue, d_blueBlurred, 
											numRows, numCols, d_filter, filterWidth);
	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }
  
  std::cout<<"blurred separated channels!"<<std::endl;
  
  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  std::cout<<"recombined channels!"<<std::endl;

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
