import sys
import fileinput
import statistics
import math
from parse import parse


if __name__ == '__main__':
    SYS_ARGV_LENGTH = 4
    assert(len(sys.argv) == SYS_ARGV_LENGTH)
    _, nodes, expected_min_cut, measure_full_contraction = sys.argv
    expected_min_cut = int(expected_min_cut)
    measure_full_contraction = False if int(measure_full_contraction) == 0 else True

    lines = iter(fileinput.input(sys.argv[SYS_ARGV_LENGTH:]))

    filename_raw = next(lines)
    filename = parse('filename: {}', filename_raw)[0]
    filename = filename.split('/')[-1]

    k_raw = next(lines)
    k = parse('k: {}', k_raw)[0]
    k = int(k)

    if measure_full_contraction:
        full_contraction_times = [0] * k
        for i in range(k):
            full_contraction_raw = next(lines)
            full_contraction = parse('full_contraction: {}', full_contraction_raw)[0]
            full_contraction_times[i] = int(full_contraction)

        full_contraction_mean = math.floor(statistics.mean(full_contraction_times))

    min_cut_raw = next(lines)
    min_cut = parse('min_cut: {}', min_cut_raw)[0]

    program_time_raw = next(lines)
    program_time = parse('program_time: {}', program_time_raw)[0]

    discovery_time_raw = next(lines)
    discovery_time = parse('discovery_time: {}', discovery_time_raw)[0]

    csv_line = f'{filename};{expected_min_cut};{nodes};{k};{min_cut};{program_time};{discovery_time}'
    if measure_full_contraction:
        csv_line = f'{csv_line};{full_contraction_mean}'

    print(csv_line)
