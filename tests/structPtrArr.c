// RUN: %RunC 'main' | FileCheck --allow-empty %s
// CHECK-NOT: {{.+}}

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
    if(argc > 1) {
        return 1;
    }
    a as[2];

    a* asPtr = as;

    asPtr[argc].a = 10;
    /*
     TODO these are still wrong, working on it
    asPtr[argc + 1].b = 20;

    asPtr[argc].inner.c = 30;
    asPtr[argc + 1].inner.d = 40;

    if (asPtr[argc].a != 10 || asPtr[argc + 1].b != 20 || asPtr[argc].inner.c != 30 || asPtr[argc + 1].inner.d != 40) {
        return 1;
    }
    */

    return 0;
}
