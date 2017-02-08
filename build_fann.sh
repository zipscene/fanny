#!/bin/bash

# This script builds and installs FANN in the node module directory

FANN_GIT_REPO="git@github.com:crispy1989/fann.git"

REALPATH="`realpath "$0"`"
ROOT="`dirname "$REALPATH"`"

echo $ROOT

# Check if it looks like FANN is already built
if [ -e "$ROOT/fann/lib/libfloatfann.so" ]; then
	echo 'FANN already built.  Skipping.'
	exit
fi

echo 'Building FANN'

# Create source directory
mkdir -p "$ROOT/fann/src"
cd "$ROOT/fann/src"

# Download the source
if [ ! -e ./fann ]; then
	git clone "$FANN_GIT_REPO" fann
	if [ $? -ne 0 ]; then
		echo 'Error cloning FANN'
		exit 1
	fi
fi
cd fann

cmake "-DCMAKE_INSTALL_PREFIX=$ROOT/fann" .
if [ $? -ne 0 ]; then
	echo 'Error using cmake to configure FANN'
	exit 1
fi

make clean all
if [ $? -ne 0 ]; then
	echo 'Error building FANN'
	exit 1
fi

make install
if [ $? -ne 0 ]; then
	echo 'Error installing FANN'
	exit 1
fi

echo 'FANN built'
exit

