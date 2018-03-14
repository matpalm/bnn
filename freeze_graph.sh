#!/usr/bin/env bash
python3 -m tensorflow.python.tools.freeze_graph \
 --input_graph bnn_graph.predict.pbtxt \
 --input_checkpoint ckpts/r19_bfs4/20180308_193817 \
 --output_node_names train_test_model/output \
 --output_graph bnn_graph.predict.frozen.pb
