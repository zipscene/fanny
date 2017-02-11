#!/bin/bash

./rebuild_all.sh debug
# gdb -ex 'run cascadetest.js' node
# node cascadetest.js > /tmp/cascadeout.txt
node cascadetest.js

