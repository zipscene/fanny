#!/bin/bash

./rebuild_all.sh debug
gdb -ex 'run cascadetest.js' node

