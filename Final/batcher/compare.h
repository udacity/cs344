int compare(float *h_out, float *h_sorted, int ARRAY_SIZE)
{	
	int failure = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        if (h_out[i] != h_sorted[i]) {
            printf("Oops! Index %i is %f, should be %f\n",
                   i, h_out[i], h_sorted[i]);
            failure = 1;
        }
    }

    if (failure == 0){
    	printf("Success! Your bitonic sort worked.");
    }

    return failure;
}