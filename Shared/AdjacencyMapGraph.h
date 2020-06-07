#pragma once

#include <algorithm>      // std::transform
#include <numeric>        // std::accumulate
#include <unordered_map>  // std::unordered_map
#include <unordered_set>  // std::unordered_set
#include <vector>         // std::vector

#include "Edge.h"
#include "random_generator.h"
#include "shared_utils.h"

/**
 * Hash functors for custom types
 */
namespace hash {
    // commutative hash functor for edges
    struct edge_hash {
        std::size_t operator()(const Edge& edge) const noexcept {
            constexpr auto hash_max = std::numeric_limits<size_t>::max();
            const auto& [i, j] = edge;
            return (i * j + (i * i) * (j * j) + (i * i * i) * (j * j * j)) % hash_max;
        }
    };
}  // namespace hash

/**
 * Adjacency Map class for undirected connected graphs.
 * Graph are assumed to be simple when created, but may evolve into multigraphs (without self-loops)
 * whenever the contract() method is called.
 */
class AdjacencyMapGraph {
    using vertex_set_t = std::unordered_set<size_t>;

    // key-value map where (key: node, value: adjacent nodes)
    using adj_map_t = std::unordered_map<size_t, vertex_set_t>;

    // key-value map where (key: edge (u, v), value: number of times the edge appears).
    // (u, v) and (v, u) are considered the same edge.
    using edge_count_map_t = std::unordered_map<Edge, size_t, hash::edge_hash>;

    /**
     * Map that stores the graph vertexes as keys and the (vertex, weight) pair connected to each
     * key as values.
     */
    adj_map_t adj_map;

    /**
     * Track the list of edges. The same edge can appear multiple times.
     */
    edge_count_map_t edge_count_map;

    /**
     * Initializes the graph starting from a map of vectors.
     * Time: O(n + m)
     * Space: O(n + m)
     */
    void init(std::unordered_map<size_t, std::vector<size_t>>&& adj_map_vec) noexcept;

public:
    /**
     * Creates a graph representation starting from a vector of edges.
     * It is assumed that the nodes v are in the range 0 <= v < n_vertex.
     * Time: O(n + m)
     * Space: O(n + m)
     */
    explicit AdjacencyMapGraph(
        std::unordered_map<size_t, std::vector<size_t>>&& adj_map_vec) noexcept :
        adj_map(adj_map_vec.size()) {
        init(std::move(adj_map_vec));
    }

    /**
     * Return the number of vertexes stored.
     * Time:  O(1)
     * Space: O(1)
     */
    [[nodiscard]] size_t size() const noexcept;

    /**
     * Return the number of edges stored.
     * Time:  O(m)
     * Space: O(m)
     */
    [[nodiscard]] size_t edge_size() const noexcept;

    /**
     * Return the list of vertexes.
     * Time:  O(n)
     * Space: O(n)
     */
    [[nodiscard]] std::vector<size_t> get_vertexes() const noexcept;

    /**
     * Return the set of vertexes adjacent to the given vertex.
     * Time:  O(1)
     * Space: O(1)
     */
    [[nodiscard]] std::unordered_set<size_t> adjacent_vertexes(size_t vertex) const noexcept;

    /**
     * Return a random edge.
     * Time:  O(m)
     * Space: O(1)
     */
    [[nodiscard]] const Edge get_random_edge() const noexcept;

    /**
     * Contract the graph deleting the edges between contractor and incorporator and moving every
     * other edge incident to contracted to incorporator.
     */
    void contract(size_t contracted, size_t incorporator);
};

inline void AdjacencyMapGraph::init(
    std::unordered_map<size_t, std::vector<size_t>>&& adj_map_vec) noexcept {
    for (auto [k, adjacent_nodes] : adj_map_vec) {
        auto& set = adj_map[k];
        set.reserve(adjacent_nodes.size());

        for (auto u : adjacent_nodes) {
            set.insert(u);
            edge_count_map[{k, u}] = 1;
        }
    }
}

inline size_t AdjacencyMapGraph::size() const noexcept {
    return adj_map.size();
}

inline size_t AdjacencyMapGraph::edge_size() const noexcept {
    std::vector<size_t> values;
    values.reserve(2 * edge_count_map.size());

    // copy the values of edge_set into values
    std::transform(edge_count_map.cbegin(), edge_count_map.cend(), std::back_inserter(values),
                   [](const auto& entry) {
                       return entry.second;
                   });

    // return the sum of the values
    return std::accumulate(values.cbegin(), values.cend(), 0);
}

inline std::vector<size_t> AdjacencyMapGraph::get_vertexes() const noexcept {
    std::vector<size_t> vertexes;
    vertexes.reserve(adj_map.size());

    // copy adj_map keys into the vertexes vector
    std::transform(adj_map.cbegin(), adj_map.cend(), std::back_inserter(vertexes),
                   [](const auto& map_entry) {
                       return map_entry.first;
                   });

    return vertexes;
}

inline std::unordered_set<size_t> AdjacencyMapGraph::adjacent_vertexes(size_t vertex) const
    noexcept {
    return adj_map.at(vertex);
}

inline const Edge AdjacencyMapGraph::get_random_edge() const noexcept {
    using namespace random_generator;
    IntegerRandomGenerator random_edge(0, edge_count_map.size() - 1);

    // extract a random entry from edge_set. Worst case: linear
    const auto& [edge, _] = *(std::next(edge_count_map.cbegin(), random_edge()));

    return edge;
}

inline void AdjacencyMapGraph::contract(size_t contracted, size_t incorporator) {
    // delete every arc between contracted and incorporator.
    // One is the selected arc, the other ones would become self-loops, so we get rid of them.
    edge_count_map.erase({contracted, incorporator});
    adj_map[incorporator].erase(contracted);
    adj_map[contracted].erase(incorporator);

    // transfer every edge incident in contracted to incorporator.
    // Note that incorporator won't appear in contracted's adjacent nodes.
    for (const auto node : adj_map.at(contracted)) {
        // replace every arc (contracted, node) with a new (incorporator, node)
        adj_map[node].erase(contracted);
        adj_map[node].insert(incorporator);
        adj_map[incorporator].insert(node);

        const size_t n_multi_edge = edge_count_map.at({contracted, node});
        edge_count_map.erase({contracted, node});
        edge_count_map[{incorporator, node}] += n_multi_edge;
    }

    // delete contracted node
    adj_map.erase(contracted);
}
