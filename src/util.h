#pragma once

#include <concepts>
#include <map>
#include <string>
#include <err.h>
#include <sys/mman.h>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wcomment"
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#pragma GCC diagnostic pop


// macros etc.
using namespace std::literals::string_literals;

#define STRINGIZE(x) #x
#define STRINGIZE_MACRO(x) STRINGIZE(x)

#ifndef NDEBUG

// __FILE__ isn't necessarily just the file name
// from https://stackoverflow.com/a/38237385
constexpr const char* file_name(const char* path) {
    const char* file = path;
    while (*path) {
        if (*path++ == '/') {
            file = path;
        }
    }
    return file;
}

#define DEBUGLOG(x)                                                                                                          \
    do {                                                                                                                     \
        if(ArgParse::args.debug()){                                                                                          \
            llvm::errs() << "File: " << file_name(__FILE__) << "\tLine " << STRINGIZE_MACRO(__LINE__) << ":\t" << x << "\n"; \
            fflush(stderr);                                                                                                  \
        }                                                                                                                    \
    } while(0)

#define IFDEBUG(x) x
#define IFDEBUGELSE(x, y) x

#else

#define DEBUGLOG(x, ...)
#define IFDEBUG(x)
#define IFDEBUGELSE(x, y) y

#endif

// exit status 2 for 2-do :)
#define EXIT_TODO_X(x) \
    do {errx(2, "TODO(File: %s\tLine " STRINGIZE_MACRO(__LINE__) "): " x "\n", file_name(__FILE__)); llvm_unreachable("");} while(0)

#define EXIT_TODO EXIT_TODO_X("Not implemented yet.")

namespace termcolor{
    extern const char* red, *green, *yellow, *blue, *magenta, *cyan, *white, *reset;

    template<typename T>
    struct colored_text{
        const char* color;
        const T& t;

        // overload for llvm::raw_ostream
        friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const colored_text& ct){
            os << ct.color << ct.t << reset;
            return os;
        }
    };

    template<typename T>
    inline colored_text<T> make(const char* color, const T& t){
        return {color, t};
    }
}

// === concepts ===

// std::sortable doesn't seem to be the same
template<typename T>
concept ordered_key = requires(T a, T b){
    { a < b }  -> std::convertible_to<bool>;
    { a == b } -> std::convertible_to<bool>;
};

// === datastructures ===

// like https://www.llvm.org/docs/ProgrammersManual.html#dss-sortedvectormap recommends, use a sorted vector for strict insert then query map
template<ordered_key K, typename V>
struct InsertBeforeQueryMap{
    using ElemPairType = typename std::pair<K,V>;

    static auto compare(const ElemPairType& elem1, const ElemPairType& elem2){
        return elem1.first < elem2.first;
    }

    llvm::SmallVector<ElemPairType> vec;

    InsertBeforeQueryMap() = default;
    InsertBeforeQueryMap(const llvm::ArrayRef<ElemPairType> &arr) : vec(arr){
        finalize();
    }

    /// O(log n): only supports lookup of actually inserted items, will segfault otherwise
    const V& operator[](const K& key) const{
        auto it = findImpl(key);
        assert((it != vec.end() && it->first == key) && "Item from InsertOnceQueryAfterwardsMap not found");
        return it->second;
    }

    /// O(log n): optionally returns a reference to the value, if it exists
    std::optional<std::reference_wrapper<const V>> at(const K& key) const{
        auto it = findImpl(key);
        if(it != vec.end() && it->first == key)
            return {it->second};
        else
            return {};
    }

    /// O(log n): returns whether the key is in the map
    bool contains(const K& key) const{
        auto it = findImpl(key);
        return it != vec.end() && it->first == key;
    }

    /// O(1): inserts a new element, if the key already exists, the element is inserted anyways, which will cause non-deterministic behaviour, if finalize is not instructed to use stable sort. If finalize uses stable sort, the first inserted element will be returned upon query
    /// you *need* to call finalize() before querying, if you use this function
    void insert(const K& key, const V& value){
        vec.push_back({key, value});
    }

    /// O(n): inserts a new element, if the key already exists, it is replaced
    /// if you find yourself using this function as your main way to insert elements, you should probably use a different datastructure
    /// you *need* to call finalize() before querying, if you use this function
    void insertOrReplace(const K& key, const V& value){
        for(auto& elem : vec){
            if(elem.first == key){
                elem.second = value;
                return;
            }
        }
        // if not found:
        vec.push_back({key, value});
    }

    /// O(n log (n)): sorts the vector, for maximum performance, call this exactly once after all insertions
    /// has to be called in between any insertions and queries
    void finalize(bool stable = false){
        if (stable)
            std::stable_sort(vec.begin(), vec.end(), compare);
        else{
            std::sort(vec.begin(), vec.end(), compare);
            IFDEBUG(
                for(size_t i = 0; i < vec.size() - 1; i++)
                    assert(vec[i].first != vec[i+1].first && "InsertBeforeQueryMap contains duplicate keys without stable sort, this will cause non-deterministic behaviour");
            );
        }
    }

    // expose iterators
    typename llvm::SmallVector<ElemPairType>::iterator begin(){
        return &*vec.begin();
    }

    typename llvm::SmallVector<ElemPairType>::iterator end(){
        return &*vec.end();
    }

private:
    auto findImpl(const K& key) const{
        assert(std::is_sorted(vec.begin(), vec.end(), compare) && "InsertBeforeQueryMap not sorted");

        auto it = std::lower_bound(vec.begin(), vec.end(), ElemPairType{key,V{}}, compare); // the V{} is just a dummy value, it will be ignored
        return it;
    }

    bool containsDebug(const K& key) const{
        for(auto& elem : vec){
            if(elem.first == key)
                return true;
        }
        return false;
    }
};

// explicit instantiation to catch errors
template struct InsertBeforeQueryMap<int, llvm::SmallString<32>>;

// === argparse === 

namespace ArgParse{
    enum kind : uint32_t{
        REQUIRED = 0x1,
        FLAG = 0x2,
    };
    struct Arg{
        std::string shortOpt{""};
        std::string longOpt{""};
        uint32_t pos{0}; //if 0, no positional arg
        std::string description{""};
        uint32_t kind{0};

        // define operators necessary for ordered map
        bool operator<(const Arg& other) const{
            if(!shortOpt.empty() && !other.shortOpt.empty())
                return shortOpt < other.shortOpt;
            else
                return longOpt < other.longOpt;
        }

        bool operator==(const Arg& other) const{
            if(!shortOpt.empty() && !other.shortOpt.empty())
                return shortOpt == other.shortOpt;
            else
                return longOpt == other.longOpt;
        }

        /// returns whether the arg has a value/has been set
        bool operator()() const;

        /// returns the args value
        std::string_view operator*() const;

        bool required() const{
            return kind & REQUIRED;
        }

        bool flag() const{
            return kind & FLAG;
        }
    };

    extern InsertBeforeQueryMap<Arg, std::string> parsedArgs;
    
    // struct for all possible arguments
    const struct {
        const Arg help{      "h", "help",           0, "Show this help message and exit",                                                          FLAG};
        const Arg input{     "i", "input",          1, "Input file"                     ,                                                          REQUIRED};
        const Arg output{    "o", "output",         2, "Output file"};
        const Arg fallback{  "F", "force-fallback", 0, "Force fallback to MLIR module compilation through the LLVM toolchain",                     FLAG};
        const Arg noFallback{"n", "no-fallback",    0, "Do not fallback to MLIR module compilation through the LLVM toolchain",                    FLAG};
        const Arg print{     "p", "print",          0, "Print parts of the compilation process. Expects 'input', 'isel', 'asm', or any combination of those as an argument"};
        const Arg debug{     "d", "debug",          0, "Print maximum debug information (set llvm::DebugFlag and all -p options)",                 FLAG};
        const Arg benchmark{ "b", "benchmark",      0, "Benchmark the compiler",                                                                   FLAG};
        const Arg iterations{"",  "iterations",     0, "Number of iterations for benchmarking (default: 1)"};
        const Arg jit{       "j", "jit",            0, "JIT compile, i.e. JIT link and immediately execute the compiled code, with the given (space separated) argvs"};

        const Arg sentinel{"", "", 0, ""};

        const Arg* const all[11] = {&help, &input, &output, &fallback, &noFallback, &print, &debug, &benchmark, &iterations, &jit, &sentinel};

        // iterator over all
        const Arg* begin() const{
            return all[0];
        }

        const Arg* end() const{
            return all[(sizeof(all)/sizeof(all[0])) - 1];
        }
    } args;

    inline bool Arg::operator()() const{
        return parsedArgs.contains(*this);
    }

    inline std::string_view Arg::operator*() const{
        assert((required() || parsedArgs.contains(*this)) && "Trying to access optional argument that has not been set");
        return parsedArgs[*this];
    }

    void printHelp(const char *argv0);

    //unordered_map doesnt work because of hash reasons (i think), so just define <, use ordered
    InsertBeforeQueryMap<Arg, std::string>& parse(int argc, char *argv[]);

} // end namespace ArgParse

// === misc ===

/// read an mlir module from a file
mlir::OwningOpRef<mlir::ModuleOp> readMLIRMod(const llvm::StringRef filename, mlir::MLIRContext& ctx);

inline std::pair<uint8_t* /* buf */, uint8_t* /* bufEnd */> mmapSpace(size_t size, int finalProt){
    auto pageSize = getpagesize();
    if(size % pageSize != 0)
        return {nullptr, nullptr};

    // to not get a SIGBUS, but a normal error, use PROT_NONE and then mprotect
    uint8_t* start =  static_cast<uint8_t*>(mmap(NULL, size, PROT_NONE, MAP_PRIVATE|MAP_ANON|MAP_NORESERVE, -1, 0));
    if(start == MAP_FAILED)
        return {nullptr, nullptr};

    if(mprotect(start, size, finalProt) != 0)
        return {nullptr, nullptr};

    return {start, start + size};
}
