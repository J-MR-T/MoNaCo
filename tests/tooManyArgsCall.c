//RUN: %CRun main | FileCheck %s

#include <inttypes.h>
int main(){
    // CHECK: Hello, these are many arguments: 0 1 2 3 4 5 6 7 8
    printf("Hello, these are many arguments: %d %d %d %d %d %d %d %d %d\n", 0,1,2,3,4,5,6,7,8);

    // CHECK-NEXT: Hello, these are many arguments: 0 1 2 3 4 5 6 7 8 9
    // this requires 5 stack args, which might break alignment
    printf("Hello, these are many arguments: %d %d %d %d %d %d %d %d %d %d\n", 0,1,2,3,4,5,6,7,8,9);

    // CHECK-NEXT: Hello these are differently sized int arguments (after 6 normal ones): 0 1 2 3 4 5 6 7 8 9
    printf("Hello these are differently sized int arguments (after 6 normal ones): %d %d %d %d %d %d %hhd %hd %d %lld\n", 0,1,2,3,4,5,(int8_t)6,(int16_t)7,8,9ll);
    return 0;
}
