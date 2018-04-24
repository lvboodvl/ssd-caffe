#cd /home/soo/caffe_ssd
./build/tools/caffe train \
--solver=SE-RSSD/proto-prelu/solver.prototxt \
--weights=SE-RSSD/snapshots-prelu/se-rssd_iter_1057.caffemodel \
--gpu all
# --weights=models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \