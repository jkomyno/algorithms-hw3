#include <any>
#include <iostream>  // std::cout, std::endl
#include <string>    // std::string_literals

#include "karger_timeout.h"
#include "read_file.h"
#include "shared_utils.h"
#include "stopwatch_decorator.h"
#include "timeout.h"

int main(int argc, char** argv) {
    using namespace std::string_literals;

    if (argc != 2) {
        std::cerr << "1 argument required: filename" << std::endl;
        exit(1);
    }

    const char* filename = argv[1];

    std::cout << "filename: "s << filename << std::endl;

    // start the stopwatch
    const auto program_time_start = stopwatch::now();

    // read the graph into an adjacent map
    auto graph = read_file(filename);

    // number of iterations required by Karger's algorithm to find the min-cut with probability 1/n
    const size_t k = utils::estimate_iterations_karger(graph->size());

    std::cout << "k: "s << k << '\n';

    // the timeout is set to 2 minutes
    auto timeout_min = 2min;

    const auto [result, karger_duration] =
        timeout::with_timeout(std::move(timeout_min), stopwatch::decorator<stopwatch::us_t>(karger),
                              std::ref(graph), k, program_time_start);

    const auto& [min_cut, discovery_time, discovery_iteration] = result;

    // stop the stopwatch
    auto program_time_stop = stopwatch::now();

    // executions of the program in microseconds
    const auto program_time =
        stopwatch::duration<stopwatch::us_t>(program_time_start, program_time_stop);

    std::cout << "min_cut: "s << std::any_cast<size_t>(min_cut) << std::endl;
    std::cout << "program_time: "s << program_time << std::endl;
    std::cout << "discovery_time: "s << std::any_cast<decltype(program_time)>(discovery_time)
              << std::endl;
    std::cout << "discovery_iteration: "s << std::any_cast<size_t>(discovery_iteration)
              << std::endl;
}
