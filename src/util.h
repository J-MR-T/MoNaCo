#pragma once

#include <concepts>
#include <dlfcn.h>
#include <map>
#include <regex>
#include <string>
#include <err.h>
#include <sys/mman.h>
#include <string_view>
#include <typeinfo>
#include <numeric>

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
using namespace std::literals;

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
    do {errx(2, "TODO(" /*"File: %s\t" */ "Line " STRINGIZE_MACRO(__LINE__) "): " x "\n" /*, file_name(__FILE__)*/); llvm_unreachable("");} while(0)

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
    struct Feature{
        std::string_view name;
        std::string_view description;
        bool defaultEnabled;

        constexpr Feature() = default;
        constexpr Feature(const std::string_view name, std::string_view description, bool defaultEnabled) : name(name), description(description), defaultEnabled(defaultEnabled){}
    };

    template<unsigned N>
    struct Features{
        struct FeatureIndex{
            unsigned index;
            Feature feature;

            constexpr FeatureIndex(unsigned index, Feature feature) : index(index), feature(feature){}

            constexpr operator Feature() const{
                return feature;
            }

            constexpr operator unsigned() const{
                return index;
            }

            operator bool&() const;

            bool operator=(bool enable) const;
        };

        static constexpr unsigned size = N;

        Feature arr[N];

        constexpr Features() : arr(){}

        constexpr void insert(Feature f, unsigned pos){
            arr[pos] = f;
        }

        constexpr auto operator[](std::string_view name) const{
            unsigned i = 0;
            for(auto& f : arr){
                if(f.name == name)
                    return FeatureIndex(i, f);
                i++;
            }
            errx(EXIT_FAILURE, "Feature %s not found", name.data());
        }

        template<typename... T>
        constexpr Features(T... features) : arr(features...){
            static_assert(sizeof...(features) == N, "Feature array size mismatch");
        }

        constexpr auto begin() const{
            return &arr[0];
        }

        constexpr auto end() const{
            return &arr[N];
        }
    };

    template<typename... T> 
    Features(T&&...) -> Features<sizeof...(T)>;

    constexpr auto features = Features(
        Feature("force-fallback",    "Force fallback to MLIR module compilation through the LLVM toolchain",                         false),
        Feature("fallback",          "Fall back to MLIR module compilation through the LLVM toolchain if MoNaCo compilation failes", true),
        Feature("codegen-dce",       "Eliminate dead instructions generated in codegen",                                             true),
        Feature("unreachable-abort", "Call `abort()` on unreachable instructions instead of simply ignoring unreachables",           false)
    );

    extern std::array<bool, features.size> enabled;

    template<unsigned N>
    Features<N>::FeatureIndex::operator bool&() const{
        return enabled[index];
    }

    template<unsigned N>
    bool Features<N>::FeatureIndex::operator=(bool enable) const{
        return enabled[index] = enable;
    }

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
        const Arg help{      "h",    "help",           0, "Show this help message and exit",                                                                                     FLAG};
        const Arg input{     "i",    "input",          1, "Input file"                     ,                                                          REQUIRED};
        const Arg output{    "o",    "output",         2, "Output file"};
        const Arg print{     "p",    "print",          0, "Print parts of the compilation process. Expects 'input', 'isel', 'asm', or any combination of those as an argument"};
        const Arg debug{     "d",    "debug",          0, "Print maximum debug information (set llvm::DebugFlag and all -p options)",                                            FLAG};
        const Arg benchmark{ "b",    "benchmark",      0, "Benchmark the compiler",                                                                                              FLAG};
        const Arg iterations{"",     "iterations",     0, "Number of iterations for benchmarking (default: 1)"};
        const Arg jit{       "j",    "jit",            0, "JIT compile, i.e. JIT link and immediately execute the compiled code, with the given (space separated) argvs"};
        const Arg forceFallback{"F", "force-fallback", 0, "shorthand for adding 'force-fallback' to the list of features",                                                       FLAG};
        const Arg featuresArg{"f",   "features",       0, ([](){
            std::string str = "Comma separated list of features:\n"s;
            for(const auto& f: features)
                str += "- " + std::string{f.name} + ": " + std::string{f.description} + "(default: " + (features[f.name] ? "true" : "false" ) + ")\n";

            return str;
            })()
        };

        const Arg sentinel{"", "", 0, ""};

        const Arg* const all[11] = {&help, &input, &output, &forceFallback, &print, &debug, &benchmark, &iterations, &jit, &featuresArg, &sentinel};

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

    inline std::pair<int, char**> parseJITArgv(){
        if(!args.jit())
            errx(EXIT_FAILURE, "Trying to parse JIT argv, but no JIT argv has been set");

        // TODO this is totally ugly, but thats what Cpp gets for not including a proper string split function
        // split jit argv with spaces
        const auto jitArgvStr = std::string{*args.jit};
        std::regex regexz("[ ]+");
        // static to avoid dangling pointer
        static std::vector<std::string> split(std::sregex_token_iterator(jitArgvStr.begin(), jitArgvStr.end(), regexz, -1), std::sregex_token_iterator());
        static std::vector<char*> jitArgv(split.size() + 1);
        for(unsigned i = 0; i < split.size(); i++){
            // TODO this is probably UB, find out if theres a better way
            jitArgv[i] = const_cast<char*>(split[i].c_str());
        }
        jitArgv[split.size()] = nullptr;

        return {split.size(), jitArgv.data()};
    }

    inline void parseFeatures(){
        auto charRange = llvm::make_range(std::begin(*args.featuresArg), std::end(*args.featuresArg));
        auto charIndexRange = llvm::enumerate(charRange);
        unsigned lastStart = 0;

        auto handleFeature = [&](auto index){
            auto feature_strv = std::string_view{charRange.begin() + lastStart, charRange.begin() + index};
            if(feature_strv.starts_with("no-"))
                features[feature_strv.substr(3)] = false;
            else
                features[feature_strv] = true;
            lastStart = index + 1;
        };

        for(auto it = charIndexRange.begin(); it != charIndexRange.end(); ++it){
            auto [index, c] = *it;
            // look for ','
            if(c == ',')
                handleFeature(index);
        }
        handleFeature((*args.featuresArg).size());

        DEBUGLOG("Features: ");
        IFDEBUG(
            for(auto feature : features){
                DEBUGLOG(feature.name << ": " << (features[feature.name] ? "true" : "false"));
            }
        )
    }

} // end namespace ArgParse

// === misc ===

using main_t = int(*)(int, char**);

inline void* checked_dlsym(llvm::StringRef name){
    // TODO do this in a better way, without construcing a std::string in between
    // TODO add possible caching to this, and enable it via template arg or smth

    // reset previous error (TODO is this necessary?)
    dlerror();
    // TODO this is a very stupid way of getting a null terminated string from this
    void* symAddr = dlsym(NULL, name.str().c_str());
    char* err = dlerror();
    if(err != NULL){
       errx(EXIT_FAILURE, "could not find symbol %s in current process, dlerror: %s", name.str().c_str(), err);
    }
    return symAddr;
}

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


inline void memcpyToLittleEndianBuffer(void* bufStart, std::integral auto value, size_t size = 0){
     if(size == 0)
        size = sizeof(value);
    // all x86(-64) instructions are little endian, but in accessing allocationSize, we have to take care of endianness

    if constexpr(std::endian::native == std::endian::little){
        // little endian, so we can just copy the bytes
        memcpy(bufStart, &value, size); // TODO can i do this in a more C++-y way?
    }else{
        static_assert(std::endian::native == std::endian::big, "endianness is neither big nor little, what is it then?");

        // big endian, so we have to reverse the bytes
        // this is wrong, seems that std::copy/reverse copy access the memory at +4, which is UB
        //std::reverse_copy(&value, &value+4, start);
        for(size_t i = 0; i < size; i++){
            static_cast<uint8_t*>(bufStart)[i] = static_cast<uint8_t*>(&value)[size - i - 1];
        }
    }
}

[[nodiscard]] inline bool fitsInto32BitImm(std::signed_integral auto val){
    return std::numeric_limits<int32_t>::min() <= val && val <= std::numeric_limits<int32_t>::max();
}

[[nodiscard]] inline bool fitsInto32BitImm(std::unsigned_integral auto val){
    return std::numeric_limits<uint32_t>::min() <= val && val <= std::numeric_limits<uint32_t>::max();
}

[[nodiscard]] inline bool iff(bool a, bool b){
    return (a && b) || (!a && !b);
}

[[nodiscard]] inline bool implies(bool a, bool b){
    return !a || b;
}
