int compare(float* h_in, float* h_out, float* h_out_shared, float* h_cmp, int ARRAY_SIZE){
    int failure = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        if (h_out[i] != h_cmp[i]) {
            fprintf(stderr, "ERROR: h_in[%d] is %f, h_out[%d] is %f, h_cmp[%d] is %f\n",
                    i, h_in[i], i, h_out[i], i, h_cmp[i]);
            failure = 1;
        }
        if (h_out_shared[i] != h_cmp[i]) {
            fprintf(stderr, "ERROR: h_in[%d] is %f, h_out_shared[%d] is %f, h_cmp[%d] is %f\n",
                    i, h_in[i], i, h_out_shared[i], i, h_cmp[i]);
            failure = 1;
        }
    }

    if (failure == 0)
    {
        printf("Success! Your smooth code worked!\n");
    }

    return failure;
}