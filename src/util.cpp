#include "util.h"

#include <regex>
#include <stdlib.h>
#include <fcntl.h>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wcomment"
#include <llvm/Config/llvm-config.h>

#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include <mlir/Parser/Parser.h>
#pragma GCC diagnostic pop

namespace termcolor{
    const char* red     = "\033[0;31m";
    const char* green   = "\033[0;32m";
    const char* yellow  = "\033[0;33m";
    const char* blue    = "\033[0;34m";
    const char* magenta = "\033[0;35m";
    const char* cyan    = "\033[0;36m";
    const char* white   = "\033[0;37m";
    const char* reset   = "\033[0m";
}

namespace ArgParse{

InsertBeforeQueryMap<Arg, std::string> parsedArgs{};

std::array<bool, features.size> enabled = ([](){
    std::array<bool, features.size> arr;
    for(unsigned i = 0; i < features.size; i++)
        arr[i] = features.arr[i].defaultEnabled;

    return arr;
})();


void printHelp(const char *argv0) {
    auto colorRed = [](const char *str) {
        std::string red = "\x1b[0;31m";
        std::string reset = "\x1b[0m";
        return red+str+reset;
    };

    llvm::outs() << colorRed("MoNaCo") << ": " << colorRed("M") << "LIR t"<<colorRed("o")<<" "<< colorRed("Na") << "tive "<< colorRed("Co") << "mpiler\n";
    llvm::errs() << "Usage: \n";
    for (auto &arg : args) {
        llvm::errs() << "  ";
        if (arg.shortOpt != "")
            llvm::errs() << "-" << arg.shortOpt;

        if (arg.longOpt != "") {
            if (arg.shortOpt != "")
                llvm::errs() << ", ";

            llvm::errs() << "--" << arg.longOpt;
        }

        if (arg.pos != 0)
            llvm::errs() << " (or positional, at position " << arg.pos << ")";
        else if (arg.flag())
            llvm::errs() << " (flag)";

        llvm::errs() << "\n    "
            // string replace all \n with \n \t here
            << std::regex_replace(arg.description, std::regex("\n"), "\n    ")
            << "\n";
    }

    llvm::errs() << "\nExamples: \n"
        << "  " << argv0 << " -i input.mlir\n"
        << "  " << argv0 << " input.mlir -fno-codegen-dce\n"
        << "  " << argv0 << " -ffallback,no-force-fallback,unreachable-abort\n";
}

InsertBeforeQueryMap<Arg, std::string>& parse(int argc, char *argv[]) {
    using std::string;

    // REFACTOR this arg string generation is not very nice
    std::stringstream ss;
    string separator = "\xff";
    ss << separator;
    for (int i = 1; i < argc; ++i) {
        ss << argv[i] << separator;
    }

    string argString = ss.str();

    // find all positional args, put them into a vector, then match them to the possible args
    std::vector<string> parsedPositionalArgs{};
    for (int i = 1; i < argc; ++i) {
        for (const auto &arg : args) {
            auto lastArg = string{argv[i - 1]};
            if (!arg.flag() && (("-" + arg.shortOpt) == lastArg||
                        ("--" + arg.longOpt) == lastArg)) {
                // the current arg is the value to another argument, so we dont count it
                goto cont;
            }
        }

        if (argv[i][0] != '-') {
            // now we know its a positional arg
            parsedPositionalArgs.emplace_back(argv[i]);
        }
cont:
        continue;
    }

    bool missingRequired = false;

    // long/short/flags
    for (const auto &arg : args) {
        if (!arg.flag()) {
            std::regex matchShort{separator + "-" + arg.shortOpt + "" + separator + "*([^" + separator + "]+)"};
            std::regex matchLong{separator + "--" + arg.longOpt + "(" + separator + "*|=)([^" + separator + "=]+)"};
            std::smatch match;
            if (arg.shortOpt != "" &&
                    std::regex_search(argString, match, matchShort)) {
                parsedArgs.insert(arg, match[1]);
            } else if (arg.longOpt != "" &&
                    std::regex_search(argString, match, matchLong)) {
                parsedArgs.insert(arg, match[2]);
            }
        } else {
            std::regex matchFlagShort{separator + "-[a-zA-z]*" + arg.shortOpt};
            std::regex matchFlagLong{separator + "--" + arg.longOpt};
            if (std::regex_search(argString, matchFlagShort) ||
                    std::regex_search(argString, matchFlagLong)) {
                parsedArgs.insert(arg, ""); // empty string for flags, will just be checked using .contains
            }
        };
    }

    // because positional args are inserted last, their manually supplied form will be preferred during querying

    // positional args
    for (const auto& arg : args)
        if (arg.pos != 0 && 
            // this is a positional arg
            parsedPositionalArgs.size() > arg.pos - 1)
                parsedArgs.insert(arg, parsedPositionalArgs[arg.pos - 1]);

    parsedArgs.finalize(true);

    for(const auto& arg : args)
        if (arg.required() && !parsedArgs.contains(arg)) {
            llvm::errs() << "Missing required argument: -" << arg.shortOpt << "/--" << arg.longOpt << "\n";
            missingRequired = true;
        }

    if (missingRequired) {
        printHelp(argv[0]);
        exit(EXIT_FAILURE);
    }
    return parsedArgs;
}

} // end namespace ArgParse


/// read an mlir module from a file
mlir::OwningOpRef<mlir::ModuleOp> readMLIRMod(const llvm::StringRef filename, mlir::MLIRContext& ctx){
    auto readOp = mlir::parseSourceFile(filename, mlir::ParserConfig(&ctx));

    if(!readOp){
        llvm::errs() << "Could not read mlir module from file " << filename << "\n";
        exit(EXIT_FAILURE);
    }

    auto module = mlir::dyn_cast<mlir::ModuleOp>(readOp.release());
    if(!module){
        llvm::errs() << "Top level operation in file " << filename << " is not a module\n";
        exit(EXIT_FAILURE);
    }

    return module;
}
