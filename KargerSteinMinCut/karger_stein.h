#pragma once

#include <algorithm>  // std::min
#include <cmath>      // std::sqrt, std::log, std::floor
#include <limits>     // std::numeric_limits
#include <memory>     // std::shared_ptr, std::unique_ptr, std::make_unique
#include <tuple>      // std::make_tuple

#include "AdjacencyMapGraph.h"
#include "full_contraction.h"
#include "shared_utils.h"

/**
 * Run Karger's randomized min-cut algorithm k times on the given graph.
 */
[[nodiscard]] size_t karger(const std::shared_ptr<AdjacencyMapGraph>& graph, size_t k) noexcept {
    // keeps track of the minimum cut
    size_t min_cut = std::numeric_limits<size_t>::max();

    // execute full_contraction k times to hopefully find the minimum cut.
    for (size_t i = 0; i < k; ++i) {
        // get the contracted graph and the full_contraction execution time in microseconds
        const auto& contracted_graph = full_contraction(graph);

        // the obtained cut is the number of edges remained in the contracted graph
        min_cut = std::min(min_cut, contracted_graph->edge_size());
    }

    return min_cut;
}

/**
 * Probability of success >= (1 / logn)
 */
[[nodiscard]] size_t fast_min_cut(const std::shared_ptr<AdjacencyMapGraph>& graph) noexcept {
    const size_t n = graph->size();

    if (n <= 6) {
        const size_t k = utils::estimate_iterations_karger(n);
        return karger(graph, k);
    }

    const size_t t = static_cast<size_t>(std::ceil((static_cast<double>(n) / std::sqrt(2))));

    auto g1 = full_contraction(graph, t);
    auto g2 = full_contraction(graph, t);

    return std::min(fast_min_cut(std::move(g1)), fast_min_cut(std::move(g2)));
}

/**
 * Run Karger's randomized min-cut algorithm k times on the given graph.
 * Return the a tuple containing the min-cut found and the discovery time.
 * Time: O(n^2 * log^3(n))
 * Space: O(m * log((n^2)/m))
 */
[[nodiscard]] size_t karger_stein(const std::shared_ptr<AdjacencyMapGraph>& graph,
                                  size_t k) noexcept {
    // keeps track of the minimum cut
    size_t min_cut = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < k; ++i) {
        // the obtained cut is the number of edges remained in the contracted graph
        min_cut = std::min(min_cut, fast_min_cut(graph));
    }

    return min_cut;
}
