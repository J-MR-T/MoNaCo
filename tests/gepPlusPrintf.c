// RUN: %CRun "abc def ghi jkl" | FileCheck %s
#include<stdio.h>

// CHECK: as_size = 2. argc: 4
// CHECK-NEXT: argv[0]: abc
// CHECK-NEXT: argv[1]: def
// CHECK-NEXT: argv[2]: ghi
// CHECK-NEXT: argv[3]: jkl
// CHECK-NEXT: as[0].a = 10
// CHECK-NEXT: as[1].b = 20
// CHECK-NEXT: as[0].inner.c = 30
// CHECK-NEXT: as[1].inner.d = 40
// CHECK-NEXT: as[argc].a = 50

typedef struct {
    int a;
    long long b;
    struct {
        int c;
        int d;
    } inner;
} a;

int main(int argc, char *argv[])
{
    a as[2];
    int as_size = sizeof(as)/sizeof(a);
    printf("as_size = %d. argc: %d\n", as_size, argc);
    for(int i = 0; i < argc; i++)
        printf("argv[%d]: %s\n", i, argv[i]);

    as[0].a = 10;
    as[1].b = 20;

    as[0].inner.c = 30;
    as[1].inner.d = 40;

    if(argc >= as_size)
        argc = sizeof(as) -1 ;
    as[argc].a = 50;

    printf("as[0].a = %d\nas[1].b = %lld\nas[0].inner.c = %d\nas[1].inner.d = %d\nas[argc].a = %d\n",
            as[0].a,      as[1].b,        as[0].inner.c,      as[1].inner.d,      as[argc].a);



    return 0;
}
