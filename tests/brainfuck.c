// hello world in brainfuck:
// RUN: %RunC 'bf ++++++++[>++++[>++>+++>+++>+<<<<-]>+>->+>>+[<]<-]>>.>>---.+++++++..+++.>.<<-.>.+++.------.--------.>+.>++.' | FileCheck %s

// CHECK: {{^}}Hello World!{{$}}

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Interpreter state
struct BFState {
    // The array and the size of the array.
    size_t array_len;
    uint8_t* array;

    // Pointer to the current position, points into array..array+array_len.
    uint8_t* cur;
};

int
searchStringForBracket(char findingBracket, const char *string, int sizeOfString, int positionInString, int direction);

// Return 0 on success, and -1 in case of an error (e.g., an out-of-bounds access).
int brainfuck(struct BFState *state, const char *program) {
    //Program counter
    int programLength = 0;
    for (int i = 0; program[i] != 0; i++, programLength++);
    int pc = 0;
    while (pc < programLength) {
        switch (program[pc]) {
            case '.':
                putchar(*(state->cur));
                break;
            case ',':
                *(state->cur) = getchar();
                break;
            case '+':
                (*(state->cur))++;
                break;
            case '-':
                (*(state->cur))--;
                break;
            case '>':
                state->cur++;
                break;
            case '<':
                state->cur--;
                break;
            case '[':
                if (*(state->cur) == 0) {
                    pc = searchStringForBracket(']', program, programLength, pc, +1);
                    //To counter the pc++ at the end
                    pc--;
                }
                break;
            case ']':
                if (*(state->cur) != 0) {
                    pc = searchStringForBracket('[', program, programLength, pc, -1);
                    //To counter the pc++ at the end
                    pc--;
                }
                break;
        }
        if ((state->cur) >= (state->array + state->array_len)||state->cur<state->array) {
            //Error, the ip is out of bounds
            return -1;
        }
        pc++;
        //error while searching for bracket
        if (pc == -1) {
            return -1;
        }
    }
    return 0;
}

//direction int: -1 means left +1 means right
//returns the new position in the string, where the findingBracket can be found, or -1 when it couldn't find one
int searchStringForBracket(char findingBracket, const char *string, int sizeOfString, int positionInString,
                           int direction) {
    int depth = 0;
    char *current = (char *) (string + positionInString);
    char startingBracket = *current;
    do {
        //Prevent page fault
        if (positionInString > sizeOfString||positionInString<0) {
            return -1;
        }
        if (*current == startingBracket) {
            depth++;
        } else if (*current == findingBracket) {
            depth--;
        }
        if (depth != 0) {
            positionInString += direction;
            current = (char *) (string + positionInString);
        }
    } while (depth != 0);
    return positionInString;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <bfprog>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t array_len = 30000;
    uint8_t* array = calloc(array_len, sizeof(uint8_t));
    if (!array) {
        fprintf(stderr, "could not allocate memory\n");
        return EXIT_FAILURE;
    }

    struct BFState state = {
            .array_len = array_len,
            .array = array,
            .cur = array,
    };
    int res = brainfuck(&state, argv[1]);
    if (res) {
        puts("an error occured");
        return EXIT_FAILURE;
    }

    free(array);

    return EXIT_SUCCESS;
}

