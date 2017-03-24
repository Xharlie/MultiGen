#!/usr/bin/env sh
set -e

/Users/xharlie/caffe/build/tools/caffe train --solver=multiGen_digit_solver.prototxt  $@ #--snapshot=snapshop_iter_5289.solverstate $@
