// RUN: %CRun main | FileCheck %s
#include <inttypes.h>
#include <stdio.h>

void f(int a, int b, int c, int d, int e, int f, uint8_t g, uint16_t h, uint32_t i, uint64_t j){
    printf("Sum: %lu\n", a + b + c + d + e + f + g + h + i + j);
}

int main(void){
    f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    // CHECK: Sum: 55
}
