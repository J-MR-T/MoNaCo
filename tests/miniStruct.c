// RUN: %CRun 'main' | FileCheck --allow-empty %s
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
    a as[2];
    as[0].a = 10;
    as[1].b = 20;

    /*as[0].inner.c = 30;*/
    /*as[1].inner.d = 40;*/

    if (as[0].a != 10 || as[1].b != 20 /*|| as[0].inner.c != 30 || as[1].inner.d != 40*/) {
        return 1;
    }

    return 0;
}
