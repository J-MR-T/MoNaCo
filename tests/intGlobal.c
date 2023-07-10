// RUN: %RunC 'main' | FileCheck %s

// CHECK: 56

int gloooob = 5;

int main(int argc, char *argv[])
{
    printf("%d", gloooob);
    gloooob = 6;
    printf("%d\n", gloooob);
    return 0;
}
