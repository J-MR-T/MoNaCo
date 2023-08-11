//RUN: %RunC main | FileCheck %s

int main() {
    // CHECK: These are doubles: 0.000000 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 7.000000
    printf("These are doubles: %f %f %f %f %f %f %f %f\n", 0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0);
}
