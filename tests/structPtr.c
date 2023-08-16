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
    a anA;

    a* anAPtr = &anA;

    anAPtr->a = 10;
    anAPtr->b = 20;
    anAPtr->inner.c = 30;
    anAPtr->inner.d = 40;

    if (anAPtr->a != 10 || anAPtr->b != 20 || anAPtr->inner.c != 30 || anAPtr->inner.d != 40) {
        return 1;
    }

    return 0;
}
