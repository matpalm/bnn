1. model.py hacked to fit the largest that works (according to the conv_deconv_output_shape_wrong example in movidius_bug_reports)

patch based model at (512,512) input was...

```
input      (1, 512, 512, 3)     #786432
e1         (1, 256, 256, 8)     #524288
e2         (1, 128, 128, 16)    #262144
e3         (1, 64, 64, 32)      #131072
e4         (1, 32, 32, 64)      #65536
d1         (1, 64, 64, 32)      #131072
d2         (1, 128, 128, 16)    #262144
d3         (1, 256, 256, 8)     #524288
logits     (1, 256, 256, 1)     #65536
```

but is now, after hacking to make it look like minimal working movidus example,

```
input      (1, 512, 512, 3)     #786432
e1         (1, 255, 255, 8)     #520200
e2         (1, 127, 127, 16)    #258064
e3         (1, 63, 63, 32)      #127008
e4         (1, 31, 31, 64)      #61504
e5         (1, 15, 15, 128)     #28800
d1         (1, 31, 31, 64)      #61504
d2         (1, 63, 63, 32)      #127008
d3         (1, 127, 127, 16)    #258064
logits     (1, 127, 127, 1)     #16129
```

2. labels generated at 1/4 size  (not 1/2 size for original model)

```
./materialise_label_db.py \
 --label-db label.201802_sample.db \
 --directory labels_0.25/ \
 --width 768 --height 1024 --rescale 0.25
```

3. model trained to overfitting

```
./train.py \
--run r12 \
--steps 100000 \
--train-steps 1000 \
--train-image-dir sample_data/training/ \
--test-image-dir sample_data/test/ \
--label-dir sample_data/labels/ \
--no-use-batch-norm --no-use-skip-connections \
--width 768 --height 1024 \
--patch-width-height 512 --label-rescale 0.25
```

4. generate graph pbtxt

```
./generate_graph_pbtxt.py \
 --no-use-skip-connections --no-use-batch-norm \
 --width 512 --height 512 \
 --pbtxt-output bnn_graph.predict.pbtxt
```

5. freeze graph

```
python3 -m tensorflow.python.tools.freeze_graph \
 --clear_devices \
 --input_graph bnn_graph.predict.pbtxt \
 --input_checkpoint `latest_ckpt_in ckpts/r12` \
 --output_node_names "train_test_model/d4/BiasAdd" \
 --output_graph graph.frozen.pb
```

( can test frozen graph on host with ... )

```
./predict_from_frozen.py \
 --image-dir sample_data/training_512_patches \
 --graph graph.frozen.pb \
 --export-pngs centroids \
 --label-rescale 0.25 \
 --model-output-node "import/train_test_model/d4/BiasAdd:0"
```

6. compile to mvNC model

```
mvNCCompile graph.frozen.pb -s 12 -in "input_imgs" -on "train_test_model/d4/BiasAdd" -o graph.mv
```

7. run inference on ncs

* note: has to do resizing hack still

```
./predict_on_ncs.py
```

outputs `from_ncs.png`
