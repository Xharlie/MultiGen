#!/usr/bin/env sh
set -e

/Users/xharlie/caffe/build/tools/caffe train --solver=multiGen_face_solver_small.prototxt  $@ #--snapshot=snapshop_iter_5289.solverstate $@
