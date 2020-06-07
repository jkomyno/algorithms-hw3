#pragma once

#include <memory>  // std::shared_ptr, std::unique_ptr, std::make_unique

#include "AdjacencyMapGraph.h"

/**
 * Contracts a copy of the given graph until it only has 2 vertexes.
 * Returns the remaining number of edges.
 * Time: O(n^2)
 * Space: O(n + m)
 */
[[nodiscard]] auto full_contraction(const std::shared_ptr<AdjacencyMapGraph>& graph,
                                    const size_t min_n = 2) noexcept
    -> std::unique_ptr<AdjacencyMapGraph> {
    // create a new copy of the graph
    auto graph_copy = std::make_unique<AdjacencyMapGraph>(*graph.get());

    // contract until n == min_n
    while (graph_copy->size() > min_n) {
        // u, v are 2 endpoints of a random edge
        const auto [u, v] = graph_copy->get_random_edge();

        // remove the arcs (u, v) and transfer all the other edges incident in u to v
        graph_copy->contract(u, v);
    }

    // the size of the cut is the number of edges (m) remained in the multigraph
    return graph_copy;
}
