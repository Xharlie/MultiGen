#!/usr/bin/env sh
set -e

/home/xharlie/dev/caffe/build/tools/caffe train --solver=GCN_solver_large.prototxt $@ # -weights genOnlySnapshot_iter_20000.caffemodel,995_multi_snap_alex_4emo__iter_30000.caffemodel $@
