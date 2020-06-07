CXX=g++-9
CXXFLAGS=-O3 -Wall -Wextra -std=c++17 -I Shared -Wno-return-type
MAINFILE=main.cpp

KARGER_MIN_CUT=KargerMinCut
KARGER_STEIN_MIN_CUT=KargerSteinMinCut

OUT_DIR="."
EXT=".out"

all: ensure_build_dir algs

algs: ${KARGER_MIN_CUT}

${KARGER_MIN_CUT}:
	${CXX} ${CXXFLAGS} "${KARGER_MIN_CUT}/${MAINFILE}" -o "${OUT_DIR}/${KARGER_MIN_CUT}${EXT}"

${KARGER_STEIN_MIN_CUT}:
	${CXX} ${CXXFLAGS} "${KARGER_STEIN_MIN_CUT}/${MAINFILE}" -o "${OUT_DIR}/${KARGER_STEIN_MIN_CUT}${EXT}"

ensure_build_dir:
	mkdir -p ${OUT_DIR}

# report:
# 	cd report; make pdf1

.PHONY: all algs ensure_build_dir clear
.PHONY: ${KARGER_MIN_CUT} ${KARGER_STEIN_MIN_CUT}

clear:
	rm "${KARGER_MIN_CUT}${EXT}" "${KARGER_STEIN_MIN_CUT}${EXT}"
