// hello world in brainfuck:
// RUN: %RunC 'bf ++++++++[>++++[>++>+++>+++>+<<<<-]>+>->+>>+[<]<-]>>.>>---.+++++++..+++.>.<<-.>.+++.------.--------.>+.>++.' | FileCheck %s

// CHECK: {{^}}Hello World!{{$}}

#include <err.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Interpreter state
struct BFState {
    // The array and the size of the array.
    size_t array_len;
    uint8_t* array;
    int* bracketInfoPcToPcMap;

    // Pointer to the current position, points into array..array+array_len.
    uint8_t* cur;
};

int
findClosingBracket(const char*, int, int);

// Return 0 on success, and -1 in case of an error (e.g., an out-of-bounds access).
int brainfuck(struct BFState *state, const char *program) {
    //Program counter
    int programLength = 0;
    int* bracketInfoPcToPcMap = state->bracketInfoPcToPcMap;
    for (int i = 0; program[i] != 0; i++, programLength++)
        bracketInfoPcToPcMap[i] = -1;

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
                if(bracketInfoPcToPcMap[pc] == -1){
                    // haven't seen this bracket yet
                    int closeBracketPos = bracketInfoPcToPcMap[pc] = findClosingBracket(program, programLength, pc);
                    // also set the other bracket
                    bracketInfoPcToPcMap[closeBracketPos] = pc;
                }

                // set the bracket
                if (*(state->cur) == 0)
                    pc = bracketInfoPcToPcMap[pc];
                break;
            case ']':
                if (*(state->cur) != 0) {
                    if(bracketInfoPcToPcMap[pc] != -1){
                        pc = bracketInfoPcToPcMap[pc];
                    }else{
                        // this should never happen: In this case, we have found a bracket that has no corersponding bracket, because the corresponding bracket would have written this to the map.
                        return -1;
                    }
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
//returns the new position in the string, where the ']' can be found, or -1 when it couldn't find one
int findClosingBracket(const char *string, int sizeOfString, int positionInString) {
    int depth = 0;
    const char *current = string + positionInString;
    char startingBracket = *current;
    static const char closingBracket = ']';
    do {
        //Prevent page fault
        if (positionInString > sizeOfString||positionInString<0) {
            return -1;
        }
        if (*current == startingBracket) {
            depth++;
        } else if (*current == closingBracket) {
            depth--;
        }
        if (depth != 0) {
            current = string + ++positionInString;
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

    int* bracketInfoPcToPcMap = malloc(sizeof(int)*array_len);
    if (!bracketInfoPcToPcMap) {
        fprintf(stderr, "could not allocate memory\n");
        return EXIT_FAILURE;
    }


    struct BFState state = {
            .array_len = array_len,
            .array = array,
            .bracketInfoPcToPcMap = bracketInfoPcToPcMap,
            .cur = array,
    };
    int res = brainfuck(&state, argv[1]);
    free(array);
    free(bracketInfoPcToPcMap);

    if (res) {
        puts("an error occured");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

