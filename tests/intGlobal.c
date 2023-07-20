// RUN: %RunC 'main' | FileCheck %s
// RUN: %RunC 'main' -pasm 2>&1 | FileCheck --ignore-case --check-prefix ASM %s

// ASM-NOT: warning: global gloooob is not at the expected address
// ASM: warning: global printf is not at the expected address{{.*}}could be external

int gloooob = 5;

int main(int argc, char *argv[])
{
    printf("%d", gloooob);
    gloooob = 6;
    printf("%d\n", gloooob);
// CHECK: 56
    return 0;
}
