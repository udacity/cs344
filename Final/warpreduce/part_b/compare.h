int compare(unsigned int h_out_warp, int sum){
 	int failure = 0;
    if (h_out_warp != sum) {
        fprintf(stderr, "GPU warp sum %d does not match expected sum %d\n", 
                h_out_warp, sum);
        failure = 1;
    }

    if (failure == 0)
    {
        printf("Success! Your warp reduce worked.\n");
    }
    else{
    	printf("Error! Your warp reduce code's output did not match sum.\n");	
    }

    return failure;
}