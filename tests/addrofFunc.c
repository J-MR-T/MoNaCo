// RUN: %RunC main | FileCheck %s
#include <stdio.h>

void test(void){
    puts("Hello!");
}

int main(int argc, char *argv[]) {
    void (*testFunc)(void) = &test;
    // CHECK: testFunc: 0x{{[0-9a-f]+}}
    printf("testFunc: %p\n", testFunc);
    // CHECK-NEXT: Hello!
    testFunc();
    return 0;
}
