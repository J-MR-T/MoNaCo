// RUN: %CRun main | FileCheck --match-full-lines %s
#include<stdio.h>

void v(int a){
    char* aPtr = &a;
    char* a2Ptr = aPtr + 2;
    putchar(*a2Ptr);
}

void v2(int a, int index){
    char* aPtr = &a;
    char* aIndexPtr = aPtr + index;
    putchar(*aIndexPtr);
}

int main(int argc, char *argv[]) {
    int a = 0x22345678;
    // should be 0x34 because little endian
    // CHECK: [[CHAR1:.]]
    putchar(0x34);
    putchar('\n');
    // CHECK-NEXT: [[CHAR1]]
    v(a);
    putchar('\n');

    for(int i=0; i<4; i++){
        putchar( a >> (8*i) );
        putchar('\n');
        v2(a, i);
        putchar('\n');
    }
    // CHECK-NEXT: [[CHAR2:.]]
    // CHECK-NEXT: [[CHAR2]]
    // CHECK-NEXT: [[CHAR3:.]]
    // CHECK-NEXT: [[CHAR3]]
    // CHECK-NEXT: [[CHAR4:.]]
    // CHECK-NEXT: [[CHAR4]]
    // CHECK-NEXT: [[CHAR5:.]]
    // CHECK-NEXT: [[CHAR5]]
    return 0;
}
