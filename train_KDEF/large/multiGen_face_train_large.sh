#!/usr/bin/env sh
set -e

/Users/xharlie/caffe/build/tools/caffe train --solver=multiGen_face_solver_large.prototxt  --snapshot=snapshop_iter_2042.solverstate $@
