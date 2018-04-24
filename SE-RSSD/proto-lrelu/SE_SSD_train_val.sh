#cd /home/soo/caffe_ssd
./build/tools/caffe train \
--solver=SE-RSSD/proto-lrelu/solver.prototxt \
--weights=models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
--gpu all
# --weights=models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \