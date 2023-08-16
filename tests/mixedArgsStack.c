// RUN: %CCheckLLVM
#include <stdio.h>

int main(void){
    // call printf with >6 int args, exactly 8 float args, and a string (well, a pointer of course), all mixed together in a random order
    printf("Hello %d %d %d %d %f %d %d %f %f %d %d %f %f %s %f %f %f\n", 1, 2, 3, 4, 4.4, 5, 6, 1.1, 2.2, 7, 8, 3.3, 5.5, "World", 6.6, 7.7, 8.8);
    return 0;
}
