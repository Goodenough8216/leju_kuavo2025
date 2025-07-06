#ifndef CPPAD_CG_ABSTRACT_C_COMPILER_INCLUDED
#define CPPAD_CG_ABSTRACT_C_COMPILER_INCLUDED
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
/* --------------------------------------------------------------------------
 *  CppADCodeGen: C++ Algorithmic Differentiation with Source Code Generation:
 *    Copyright (C) 2012 Ciengis
 *    Copyright (C) 2018 Joao Leal
 *
 *  CppADCodeGen is distributed under multiple licenses:
 *
 *   - Eclipse Public License Version 1.0 (EPL1), and
 *   - GNU General Public License Version 3 (GPL3).
 *
 *  EPL1 terms and conditions can be found in the file "epl-v10.txt", while
 *  terms and conditions for the GPL3 can be found in the file "gpl3.txt".
 * ----------------------------------------------------------------------------
 * Author: Joao Leal
 */

namespace CppAD {
namespace cg {

/**
 * Default implementation of a C compiler class used to create
 * dynamic and static libraries
 *
 * @author Joao Leal
 */
template<class Base>
class AbstractCCompiler : public CCompiler<Base> {
protected:
    std::string _path; // the path to the gcc executable
    std::string _tmpFolder;
    std::string _sourcesFolder; // path where source files are saved
    std::set<std::string> _ofiles; // compiled object files
    std::set<std::string> _sfiles; // compiled source files
    std::vector<std::string> _compileFlags;
    std::vector<std::string> _compileLibFlags;
    std::vector<std::string> _linkFlags;
    bool _verbose;
    bool _saveToDiskFirst;
public:

    AbstractCCompiler(const std::string& compilerPath) :
        _path(compilerPath),
        _tmpFolder("cppadcg_tmp"),
        _sourcesFolder("cppadcg_sources"),
        _verbose(false),
        _saveToDiskFirst(false) {
    }

    AbstractCCompiler(const AbstractCCompiler& orig) = delete;
    AbstractCCompiler& operator=(const AbstractCCompiler& rhs) = delete;

    std::string getCompilerPath() const {
        return _path;
    }

    void setCompilerPath(const std::string& path) {
        _path = path;
    }

    const std::string& getTemporaryFolder() const override {
        return _tmpFolder;
    }

    void setTemporaryFolder(const std::string& tmpFolder) override {
        _tmpFolder = tmpFolder;
    }

    bool isSaveToDiskFirst() const override {
        return _saveToDiskFirst;
    }

    void setSaveToDiskFirst(bool saveToDiskFirst) override {
        _saveToDiskFirst = saveToDiskFirst;
    }

    const std::string& getSourcesFolder() const override {
        return _sourcesFolder;
    }

    void setSourcesFolder(const std::string& srcFolder) override {
        _sourcesFolder = srcFolder;
    }

    const std::set<std::string>& getObjectFiles() const override {
        return _ofiles;
    }

    const std::set<std::string>& getSourceFiles() const override {
        return _sfiles;
    }

    const std::vector<std::string>& getCompileFlags() const {
        return _compileFlags;
    }

    void setCompileFlags(const std::vector<std::string>& compileFlags) {
        _compileFlags = compileFlags;
    }

    void addCompileFlag(const std::string& compileFlag) {
        _compileFlags.push_back(compileFlag);
    }

    const std::vector<std::string>& getLinkFlags() const {
        return _linkFlags;
    }

    void setLinkFlags(const std::vector<std::string>& linkFlags) {
        _linkFlags = linkFlags;
    }

    void addLinkFlag(const std::string& linkFlag) {
        _linkFlags.push_back(linkFlag);
    }

    const std::vector<std::string>& getCompileLibFlags() const {
        return _compileLibFlags;
    }

    void setCompileLibFlags(const std::vector<std::string>& compileLibFlags) {
        _compileLibFlags = compileLibFlags;
    }

    void addCompileLibFlag(const std::string& compileLibFlag) {
        _compileLibFlags.push_back(compileLibFlag);
    }

    bool isVerbose() const override {
        return _verbose;
    }

    void setVerbose(bool verbose) override {
        _verbose = verbose;
    }

    /**
     * Compiles the provided C source code.
     *
     * @param library the path of the dynamic library to be created
     * @param sources maps the names to the content of the source files
     * @param posIndepCode whether or not to create position-independent
     *                     code for dynamic linking
     */
    void compileSources(const std::map<std::string, std::string>& sources,
                        bool posIndepCode,
                        JobTimer* timer = nullptr) override {
        compileSources(sources, posIndepCode, timer, ".o", _ofiles);
    }

    virtual void compileSources(const std::map<std::string, std::string> &sources,
                                bool posIndepCode,
                                JobTimer *timer,
                                const std::string &outputExtension,
                                std::set<std::string> &outputFiles)
    {
        using namespace std::chrono;

        if (sources.empty())
            return; // nothing to do

        system::createFolder(this->_tmpFolder);

        // determine the maximum file name length
        size_t maxsize = 0;
        for (const auto &source : sources)
        {
            _sfiles.insert(source.first);
            std::string file = system::createPath(this->_tmpFolder, source.first + outputExtension);
            maxsize = std::max<size_t>(maxsize, file.size());
        }

        size_t countWidth = std::ceil(std::log10(sources.size()));

        if (timer != nullptr)
        {
            size_t ms = 3 + 2 * countWidth + 1 + JobTypeHolder<>::COMPILING.getActionName().size() + 2 + maxsize + 5;
            ms += timer->getJobCount() * 2;
            if (timer->getMaxLineWidth() < ms)
                timer->setMaxLineWidth(ms);
        }
        else if (_verbose)
        {
            std::cout << std::endl;
        }

        if (_saveToDiskFirst)
        {
            system::createFolder(_sourcesFolder);
        }

        // Prepare for multi-threading
        std::mutex outputMutex;
        std::atomic<size_t> count(0);
        std::vector<std::thread> threads;
        const size_t numThreads = std::thread::hardware_concurrency();

        auto compileWorker = [&](const std::pair<std::string, std::string> &source)
        {
            std::string file = system::createPath(this->_tmpFolder, source.first + outputExtension);
            // std::cout << "start job:"<<file<<std::endl;
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                outputFiles.insert(file);
            }

            std::ostringstream os;
            steady_clock::time_point beginTime;

            size_t currentCount = ++count;

            if (timer != nullptr || _verbose)
            {
                os << "[" << std::setw(countWidth) << std::setfill(' ') << std::right << currentCount
                   << "/" << sources.size() << "]";
            }

            if (timer != nullptr)
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                timer->startingJob("'" + file + "'", JobTypeHolder<>::COMPILING, os.str());
            }
            else if (_verbose)
            {
                beginTime = steady_clock::now();
                std::lock_guard<std::mutex> lock(outputMutex);
                char f = std::cout.fill();
                std::cout << os.str() << " compiling "
                          << std::setw(maxsize + 9) << std::setfill('.') << std::left
                          << ("'" + file + "' ") << " ";
                std::cout.flush();
                std::cout.fill(f); // restore fill character
            }

            std::ofstream sourceFile;
            std::string srcfile = system::createPath(_sourcesFolder, source.first);
            if (_saveToDiskFirst)
            {
                // save a new source file to disk
                std::ofstream sourceFile;
                // std::string srcfile = system::createPath(_sourcesFolder, source.first);
                sourceFile.open(srcfile.c_str());
                sourceFile << source.second;
                sourceFile.close();
            }

            // compile library from source code in memory
            // compileSource(source.second, file, posIndepCode);
            compileFile(srcfile, file, posIndepCode);

            if (timer != nullptr)
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                timer->finishedJob();
            }
            else if (_verbose)
            {
                steady_clock::time_point endTime = steady_clock::now();
                duration<float> dt = endTime - beginTime;
                std::lock_guard<std::mutex> lock(outputMutex);
                std::cout << "done [" << std::fixed << std::setprecision(3)
                          << dt.count() << "]" << std::endl;
            }
        };

        // Start threads
        for (const auto &source : sources)
        {
            if (threads.size() >= numThreads)
            {
                threads.front().join();
                threads.erase(threads.begin());
            }
            threads.emplace_back(compileWorker, source);
        }

        // Wait for all threads to finish
        for (auto &thread : threads)
        {
            thread.join();
        }
    }

    /**
     * Creates a dynamic library from a set of object files
     *
     * @param library the path to the dynamic library to be created
     */
    virtual void buildDynamic(const std::string& library,
                              JobTimer* timer = nullptr) override = 0;

    void cleanup() override {
        // clean up;
        for (const std::string& it : _ofiles) {
            if (remove(it.c_str()) != 0)
                std::cerr << "Failed to delete temporary file '" << it << "'" << std::endl;
        }
        _ofiles.clear();
        _sfiles.clear();

        remove(this->_tmpFolder.c_str());
    }

    virtual ~AbstractCCompiler() {
        cleanup();
    }

protected:

    /**
     * Compiles a single source file into an object file.
     *
     * @param source the content of the source file
     * @param output the compiled output file name (the object file path)
     */
    virtual void compileSource(const std::string& source,
                               const std::string& output,
                               bool posIndepCode) = 0;

    /**
     * Compiles a single source file into an object file.
     *
     * @param path the path to the source file
     * @param output the compiled output file name (the object file path)
     */
    virtual void compileFile(const std::string& path,
                             const std::string& output,
                             bool posIndepCode) = 0;
};

} // END cg namespace
} // END CppAD namespace

#endif
