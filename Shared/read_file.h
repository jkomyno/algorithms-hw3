#pragma once

#include <algorithm>      // std::transform
#include <fstream>        // std::fstream
#include <iterator>       // std::istream_iterator
#include <limits>         // std::numeric_limits
#include <memory>         // std::shared_ptr, std::make_shared
#include <sstream>        // std::istringstream
#include <stdexcept>      // std::exception
#include <string>         // std::string, std::string_literals
#include <unordered_map>  // std::unordered_map
#include <vector>         // std::vector

#include "AdjacencyMapGraph.h"
#include "shared_utils.h"

/**
 * Helper that reads a connected graph from a text file.
 * We assume that the nodes are labeled with a label x,
 * where 1 <= x <= n (n is the number of nodes).
 * The nodes are going to be saved into the adjacency map
 * class AdjacencyMapGraph with their label decremented by 1
 * (thus 0 <= x' <= n-1).
 * Creating a graph representation from an input file takes O(m * (n + m)) time.
 */
inline std::shared_ptr<AdjacencyMapGraph> read_file(const char* filename) {
    using namespace std::string_literals;

    std::fstream file(filename);
    if (!file.good()) {
        throw std::runtime_error("File doesn't exist"s);
    }

    // the index represent the starting nodes, the indexed vector represents the adjacent nodes
    std::unordered_map<size_t, std::vector<size_t>> graph;

    // read input file line by line
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream buffer(line);
        std::vector<size_t> nodes{(std::istream_iterator<size_t>(buffer)),
                                  std::istream_iterator<size_t>()};

        // extract the starting node from the parsed line
        const size_t head = utils::pop_front(nodes);

        graph[head] = nodes;
    }

    // release file
    file.close();

    return std::make_shared<AdjacencyMapGraph>(std::move(graph));
}
