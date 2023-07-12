// RUN: %RunC main

#include <stdlib.h>

void* nullptr_ = NULL;

int main(int argc, char *argv[])
{
    int a = 0;
    int* ptr = &a;
    if (ptr == nullptr_)
        return 1;
    
    ptr = NULL;
    if (ptr == nullptr_)
        return 0;
    else
        return 1;
}
