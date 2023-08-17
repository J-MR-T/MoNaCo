// RUN: %CCheckLLVM

float f(float a, float b){
    return a * b;
}

double d(double a, double b){
    return a * b;
}

int main(void){
    float a = 2.0;
    float b = 3.0;
    return (int)(f(a,b) + d(2.0, 3.0)) - 12;
}
