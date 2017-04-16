#!/usr/bin/env sh
set -e

/Users/xharlie/caffe/build/tools/caffe train --solver=GCN_solver_large.prototxt  $@ # -weights alex_3797.caffemodel $@
