#include <stdio.h>
#include "gputimer.h"

const int N= 1024;		// matrix size is NxN
const int K= 32;				// tile size is KxK

// Utility functions: compare, print, and fill matrices
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at: %s : %d\n", file,line);
    fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);;
    exit(1);
  }
}

int compare_matrices(float *gpu, float *ref)
{
	int result = 0;

	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
    		if (ref[i + j*N] != gpu[i + j*N])
    		{
    			// printf("reference(%d,%d) = %f but test(%d,%d) = %f\n",
    			// i,j,ref[i+j*N],i,j,test[i+j*N]);
    			result = 1;
    		}
    return result;
}

void print_matrix(float *mat)
{
	for(int j=0; j < N; j++) 
	{
		for(int i=0; i < N; i++) { printf("%4.4g ", mat[i + j*N]); }
		printf("\n");
	}	
}

// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat)
{
	for(int j=0; j < N * N; j++)
		mat[j] = (float) j;
}



void 
transpose_CPU(float in[], float out[])
{
	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
      		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched on a single thread
__global__ void 
transpose_serial(float in[], float out[])
{
	for(int j=0; j < N; j++)
		for(int i=0; i < N; i++)
			out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per row of output matrix
__global__ void 
transpose_parallel_per_row(float in[], float out[])
{
	int i = threadIdx.x;

	for(int j=0; j < N; j++)
		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per element, in KxK threadblocks
// thread (x,y) in grid writes element (i,j) of output matrix 
__global__ void 
transpose_parallel_per_element(float in[], float out[])
{
	int i = blockIdx.x * K + threadIdx.x;
	int j = blockIdx.y * K + threadIdx.y;

	out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ void 
transpose_parallel_per_element_tiled(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i  = blockIdx.x * K, in_corner_j  = blockIdx.y * K;
	int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[K][K];

	// coalesced read from global mem, TRANSPOSED write into shared mem:
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y)*N];
	__syncthreads();
	// read from shared mem, coalesced write to global mem:
	out[(out_corner_i + x) + (out_corner_j + y)*N] = tile[x][y];
}

// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ void 
transpose_parallel_per_element_tiled16(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i  = blockIdx.x * 16, in_corner_j  = blockIdx.y * 16;
	int out_corner_i = blockIdx.y * 16, out_corner_j = blockIdx.x * 16;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[16][16];

	// coalesced read from global mem, TRANSPOSED write into shared mem:
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y)*N];
	__syncthreads();
	// read from shared mem, coalesced write to global mem:
	out[(out_corner_i + x) + (out_corner_j + y)*N] = tile[x][y];
}

// to be launched with one thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array padded to avoid bank conflicts
__global__ void 
transpose_parallel_per_element_tiled_padded(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i  = blockIdx.x * K, in_corner_j  = blockIdx.y * K;
	int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[K][K+1];

	// coalesced read from global mem, TRANSPOSED write into shared mem:
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y)*N];
	__syncthreads();
	// read from shared mem, coalesced write to global mem:
	out[(out_corner_i + x) + (out_corner_j + y)*N] = tile[x][y];
}

// to be launched with one thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array padded to avoid bank conflicts
__global__ void 
transpose_parallel_per_element_tiled_padded16(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i  = blockIdx.x * 16, in_corner_j  = blockIdx.y * 16;
	int out_corner_i = blockIdx.y * 16, out_corner_j = blockIdx.x * 16;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[16][16+1];

	// coalesced read from global mem, TRANSPOSED write into shared mem:
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y)*N];
	__syncthreads();
	// read from shared mem, coalesced write to global mem:
	out[(out_corner_i + x) + (out_corner_j + y)*N] = tile[x][y];
}

int main(int argc, char **argv)
{
	int numbytes = N * N * sizeof(float);

	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	float *gold = (float *) malloc(numbytes);

	fill_matrix(in);
	transpose_CPU(in, gold);

	float *d_in, *d_out;

	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	GpuTimer timer;

/*  
 * Now time each kernel and verify that it produces the correct result.
 *
 * To be really careful about benchmarking purposes, we should run every kernel once
 * to "warm" the system and avoid any compilation or code-caching effects, then run 
 * every kernel 10 or 100 times and average the timings to smooth out any variance. 
 * But this makes for messy code and our goal is teaching, not detailed benchmarking.
 */

	timer.Start();
	transpose_serial<<<1,1>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_serial: %g ms.\nVerifying transpose...%s\n", 
	       timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

	timer.Start();
	transpose_parallel_per_row<<<1,N>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_row: %g ms.\nVerifying transpose...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

	dim3 blocks(N/K,N/K); // blocks per grid
	dim3 threads(K,K);	// threads per block

	timer.Start();
	transpose_parallel_per_element<<<blocks,threads>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element: %g ms.\nVerifying transpose...%s\n",
		   timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

	timer.Start();
	transpose_parallel_per_element_tiled<<<blocks,threads>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n", 
		   K, K, timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");
	
	dim3 blocks16x16(N/16,N/16); // blocks per grid
	dim3 threads16x16(16,16);	 // threads per block

	timer.Start();
	transpose_parallel_per_element_tiled16<<<blocks16x16,threads16x16>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled 16x16: %g ms.\nVerifying ...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");
	
	timer.Start();
 	transpose_parallel_per_element_tiled_padded16<<<blocks16x16,threads16x16>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled_padded 16x16: %g ms.\nVerifying...%s\n", 
	       timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

	cudaFree(d_in);
	cudaFree(d_out);
}