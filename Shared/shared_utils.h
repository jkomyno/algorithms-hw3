#pragma once

#include <tuple> // std::tuple

namespace utils {
    template <typename T>
    T pop_at_index(std::vector<T>& vec, const size_t i) noexcept {
        const T result = vec.front();
        const size_t size = vec.size();
        const size_t last_index = size - 1;

        std::swap(vec[i], vec[last_index]);
        vec.pop_back();

        return result;
    }

    /**
     * Remove the first element of the vector in O(1). The order of the vector isn't preserved.
     */
    template <typename T>
    T pop_front(std::vector<T>& vec) noexcept {
        return pop_at_index<T>(vec, 0);
    }

    /**
     * Return the binomial coefficient (n choose 2).
     * This particular case can be computed as (n * (n-1)) / 2.
     * If n < 2, the binomial coefficient is ill-formed, so 1 is returned.
     */
    [[nodiscard]] double choose_2(const size_t n) noexcept {
        if (n > 2) {
            return static_cast<double>(n * (n - 1)) / 2;
        }

        return 1;
    }

    /**
     * Return the number of iterations needed to have probability <= (1/n)^d that Karger's algorithm
     * returns a cut which isn't minimum.
     */
    [[nodiscard]] size_t estimate_iterations_karger(const size_t n, const size_t d = 1) noexcept {
        double iterations = d * choose_2(n) * std::log(n);

        // round it up to the next integer
        return static_cast<size_t>(std::ceil(iterations));
    }

    /**
     * Return the number of iterations needed to have probability <= (1/n) that Karger's algorithm
     * returns a cut which isn't minimum.
     */
    [[nodiscard]] size_t estimate_iterations_karger_stein(const size_t n) noexcept {
        double iterations = (n * std::log(n)) / (n - 1);

        // round it up to the next integer
        return static_cast<size_t>(std::ceil(iterations));
    }
}
