#pragma once

#include <tuple>        // std::tuple, std::tie
#include <type_traits>  // std::invoke_result_t

#include "stopwatch.h"

namespace detail {
    template <typename T>
    class return_wrapper {
        T val;

    public:
        template <typename F, typename... Args>
        return_wrapper(F&& func, Args&&... args) :
            val(std::forward<F>(func)(std::forward<Args>(args)...)) {
        }

        T&& value() {
            return std::move(val);
        }
    };

    template <>
    class return_wrapper<void> {
    public:
        template <typename F, typename... Args>
        return_wrapper(F&& func, Args&&... args) {
            std::forward<F>(func)(std::forward<Args>(args)...);
        }

        void value() {
        }
    };

    template <typename>
    struct is_tuple : std::false_type {};

    template <typename... T>
    struct is_tuple<std::tuple<T...>> : std::true_type {};
}  // namespace detail

namespace stopwatch {
    /**
     * Curried function that executes the given function and arguments and returns a tuple
     * containing the result of the function and the execution time of the same function. If the
     * result type of the function is already a tuple, the execution time is appended to the end of
     * that tuple.
     */
    template <typename TimeDuration, typename F>
    auto decorator(F&& func) {
        return [func = std::forward<F>(func)](auto&&... args) {
            // start the stopwatch
            auto start_time = stopwatch::now();

            // run the function and capture its result
            using result_t = std::invoke_result_t<F, decltype(args)...>;
            detail::return_wrapper<result_t> result(func, std::forward<decltype(args)>(args)...);

            // stop the stopwatch
            auto stop_time = stopwatch::now();

            // execution time expressed in TimeDuration
            const auto func_duration = stopwatch::duration<TimeDuration>(start_time, stop_time);

            if constexpr (detail::is_tuple<result_t>::value) {
                return std::tuple_cat(result.value(), std::tie(func_duration));
            } else {
                return std::make_tuple(result.value(), func_duration);
            }
        };
    }
}  // namespace stopwatch
