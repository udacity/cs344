#include <stdio.h>

int main(int argc,char **argv)
{   
    printf("(%g + %g) + %g == %g\n%g + (%g + %g) == %g\n", 
        1.f, 1e99, -1e99, (1.f + 1e99)+ -1e99, 
        1.f, 1e99, -1e99, 1.f + (1e99 + -1e99));
    return 0;
}