// RUN: %CRun main | FileCheck --match-full-lines %s
#include <stdio.h>

void test(void){
    puts("Hello!");
}

void test2(void);

int main(int argc, char *argv[]) {
    void (*testFunc)(void) = &test;
    // CHECK: testFunc: 0x{{[0-9a-f]+}}
    printf("testFunc: %p\n", testFunc);
    // CHECK-NEXT: Hello!
    testFunc();
    testFunc = &test2;
    // CHECK-NEXT: Hello2!
    testFunc();

    // now call putchar indirectly
    int (*putcharFunc)(int) = &putchar;
    putcharFunc('a');
    // CHECK-NEXT: a
    return 0;
}

void test2(void){
    puts("Hello2!");
}
