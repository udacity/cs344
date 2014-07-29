// error checking utility functions
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

void printArray(float in[], int N)
{
	for (int i=0; i<N; i++) { printf("%g ", in[i]); }
	printf("\n");
}

int compareArrays(float *ref, float *test, int N)
{
	// ignore the boundaries
	for (int i=2; i<N-2; i++)
	{
		if (ref[i] != test[i]) 
		{
			printf("Error: solution does not match reference!\n");
			printf("first deviation at location %d\n", i);
			printf("reference array:\n"); printArray(ref, N);
			printf("solution array:\n"); printArray(test, N);
			return 1;
		}
	}
	printf("Verified!\n");
	return 0;
}