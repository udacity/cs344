int compare(unsigned int h_out_shared, int sum){
 	int failure = 0;
    if (h_out_shared != sum) {
        fprintf(stderr, "GPU shared sum %d does not match expected sum %d\n", 
                h_out_shared, sum);
        failure = 1;
    }

    if (failure == 0)
    {
        printf("Success! Your shared warp reduce worked.\n");
    }
    else{
    	printf("Error! Your shared reduce code's output did not match sum.\n");	
    }

    return failure;
}