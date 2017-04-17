#!/usr/bin/env sh
set -e

/home/xharlie/dev/caffe/build/tools/caffe train --solver=GCN_solver_large.prototxt   -weights 995_multi_snap_alex_4emo__iter_30000.caffemodel $@
