#pragma once

#include <iostream>  // std::cout, std::endl
#include <limits>    // std::numeric_limits
#include <memory>    // std::shared_ptr, std::unique_ptr, std::make_unique
#include <tuple>     // std::make_tuple

#include "AdjacencyMapGraph.h"
#include "full_contraction.h"
#include "stopwatch_decorator.h"
#include "timeout.h"

/**
 * Run Karger's randomized min-cut algorithm k times on the given graph.
 * Return the a tuple containing the min-cut found and the discovery time.
 * Time: O(n^4 * log(n))
 * Space: O(n + m)
 */
[[nodiscard]] auto karger(timeout::timeout_signal& signal,
                          const std::shared_ptr<AdjacencyMapGraph>& graph, size_t k,
                          const stopwatch::time_point_t program_time_start) noexcept {
    // keep track of the min-cut discovery time
    stopwatch::time_point_t discovery_time_stop;

    // keeps track of the minimum cut
    size_t min_cut = std::numeric_limits<size_t>::max();

    bool keep_going = true;

    // execute full_contraction k times to hopefully find the minimum cut.
    for (size_t i = 0; i < k && keep_going; ++i) {
        // get the contracted graph and the full_contraction execution time in microseconds
        const auto [contracted_graph, full_contraction_duration] =
            stopwatch::decorator<stopwatch::us_t>(full_contraction)(graph, 2);

        // the obtained cut is the number of edges remained in the contracted graph
        const size_t cut = contracted_graph->edge_size();

        if (cut < min_cut) {
            // a better cut has been found, so we update min_cut and reset stop_discovery_time
            min_cut = cut;
            discovery_time_stop = stopwatch::now();
        }

        std::cout << "full_contraction: " << full_contraction_duration << '\n';

        keep_going = !signal.is_expired();
    }

    // number of microseconds needed to find the lowest cut among all k full_contraction iterations
    const auto discovery_time =
        stopwatch::duration<stopwatch::us_t>(program_time_start, discovery_time_stop);

    return std::make_tuple(min_cut, discovery_time);
}
