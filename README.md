# Algorithms-HW3

[![Build Status](https://travis-ci.com/jkomyno/algorithms-hw3.svg?token=VSm1u6swXqyzsdGeq7Kp&branch=master)](https://travis-ci.com/jkomyno/algorithms-hw3)

> Minimum Cut with Karger algorithm.

An hands-on experience about algorithms over graph and graph theory
with the focus on efficiency.

## Usage

The repositores comes with some utility commands that allows you to
compile, test and bench owr algorithms. We use _make_ to automatize
this process.

**Available commands**

- `make all`, to compile all the algorithm sources in this project.
- `make ALG`, where _ALG_ is one of _KargerMinCut_, _KargerMinCutTimeout_,
  _KargerSteinMinCut_ to compile given algorithm sources.
- `make clear`, to clean up the working directory.

Within the Makefile we provieded some variables to modify our pipeline.
In particular you can use your own compiler rewriting the `CXX` flag. Other
variables can be modified, for example `CXXFLAGS`, `OUT_DIR` and `EXT`.

Example

```
make CXX="g++" CXXFLAGS="-O3 -std=c++17 -I Shared" OUT_DIR="build" EXT="exe" all
```

**Scripts**

We created some bash scripts in order to automatize recurrent tasks
like benchmarking. You can take a look at the script `run.sh`,
`runall.sh` and `process.py` for further informations.

In particular,

- `run.sh` executes a program with some options and saves its output,
- `runall.sh`, executes all listed programs using `run.sh`,
- `process.py`, processes a program output and returns it in a machine readable format.

**Report**

We created also a report with a not-so-short description of owr work.
You can find it in the directory [report](./report). The report is a
**LaTeX project** subdivided for readability purposes in some directories
like *chapters* and *images*. All starts in the `main.tex` file.

In order to compile sources, navigate to `report` directory and
* if you have `pdflatex` installed, run `make pdf`,
* if you have `docker` installed, run `make pdf2`.

Note: we didn't tested the doker build, there could be errors.

## Project Structure

The project is structured as a unique Visual Studio solution
containing multiple subprojects, one for every implemented algorithm.
The code for each project is stored in a folder with the same name of
the related algorithm.

These projects are:

- [KargerMinCut](./KargerMinCut): minimun cut implemented with Karger algorithm;
- [KargerMinCutTimeout](./KargerMinCutTimeout): a variant of KargerMinCut with timeout;
- [KargerSteinMinCut](./KargerSteinMinCut): minumum cut implemented with Karger and Stein algorithm.

The shared data structures and utils are stored in the _Shared_ folder.

The project comes with some extra folders:

- **benchmark**: it contains CSV benchmarks of the algorithm as well
    as the script used to analyze them
    ([analysis.py](./benchmark/analysis.py));
- **datasets**: it contains the input data for the graphs given by our
    professor;

## Related Projects

* [**algorithms-hw2**](https://github.com/jkomyno/algorithms-hw2) the
previous homework on Travelling Salesman Problem with Held & Karp,
approximated and heuristic algorithms.
* [**algorithms-hw1**](https://github.com/jkomyno/algorithms-hw1) the
previous homework on MST algorithms.

## Authors

**Bryan Lucchetta**

- GitHub: [@1-coder](https://github.com/1-coder)

**Luca Parolari**

- GitHub: [@lparolari](https://github.com/lparolari)

**Alberto Schiabel**

- GitHub: [@jkomyno](https://github.com/jkomyno)

## License

This project is MIT licensed. See [LICENSE](LICENSE) file.
