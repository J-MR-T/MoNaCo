// RUN: %RunC | FileCheck %s

// CHECK: as_size = 2. argv[0]: argv0

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
    printf("as_size = %d. argv[0]: %s\n", as_size, argv[0]);
    as[0].a = 10;
    as[1].b = 20;

    as[0].inner.c = 30;
    as[1].inner.d = 40;

    if(argc >= as_size)
        argc = sizeof(as) -1 ;
    as[argc].a = 50;



    return 0;
}
