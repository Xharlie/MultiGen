#!/usr/bin/env sh
set -e

/Users/xharlie/caffe/build/tools/caffe train --solver=solver.prototxt  --snapshot=snapshop_iter_3000.solverstate $@
