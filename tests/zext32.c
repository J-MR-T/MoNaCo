// RUN: %CRun main | FileCheck %s
#include <inttypes.h>
#include <stdio.h>

void f(uint8_t g, uint16_t h, uint32_t i, uint64_t j){
    printf("Sum: %lu\n", g + h + i + j);
}

int main(void){
    f(5, 6, 7, 8);
    // CHECK: Sum: 26
}
