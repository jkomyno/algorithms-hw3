#pragma once

#include <iterator>  // std::next
#include <utility>   // std::initializer_list

/**
 * Edge represents an edge between two vertexes.
 */
class Edge {
public:
    size_t from;
    size_t to;

    explicit Edge(size_t from, size_t to) noexcept : from(from), to(to) {
    }

    Edge(std::initializer_list<size_t> l) noexcept :
        from(*(l.begin())), to(*(std::next(l.begin(), 1))) {
    }

    // (from, to) and (to, from) are considered the same edge
    bool operator==(const Edge& e) const noexcept {
        return (from == e.from && to == e.to) || (to == e.from && from == e.to);
    }

    friend void swap(Edge& lhs, Edge& rhs) {
        std::swap(lhs.from, rhs.from);
        std::swap(lhs.to, rhs.to);
    }
};
