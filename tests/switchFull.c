// RUN: %CRun main | FileCheck --match-full-lines %s

void print(int n){
    switch(n){
        case 1: puts("1!"); break;
        case 6: puts("6!"); break;
        case 2: puts("2!"); break;
        case 5: puts("5!"); break;
        case 9: puts("9!"); break;
        default: puts("other!");
    }
}

int main(void){
    for(int i=0; i<=10; i++){
        print(i);
    }
    // CHECK: 1!
    // CHECK-NEXT: 2!
    // CHECK-NEXT: other!
    // CHECK-NEXT: other!
    // CHECK-NEXT: 5!
    // CHECK-NEXT: 6!
    // CHECK-NEXT: other!
    // CHECK-NEXT: other!
    // CHECK-NEXT: 9!
    // CHECK-NEXT: other!
}
