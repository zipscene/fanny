#!/bin/bash

REALPATH="`realpath "$0"`"
ROOT="`dirname "$REALPATH"`"

GYPFLAGS=""
CMAKEFLAGS="-DCMAKE_INSTALL_PREFIX=$ROOT/fann"

$ROOT/build_fann.sh

if [ "$1" = "debug" ]; then
	GYPFLAGS="${GYPFLAGS} -d"
	CMAKEFLAGS="${CMAKEFLAGS} -DCMAKE_BUILD_TYPE=Debug"
fi

cd "$ROOT/fann/src/fann"

make clean && cmake $CMAKEFLAGS . && make && make install

cd "$ROOT"

node-gyp clean
node-gyp $GYPFLAGS configure && node-gyp $GYPFLAGS build
