#cd /home/soo/caffe_ssd
./build/tools/caffe train \
--solver=SE-RSSD/solver.prototxt \
--weights=SE-RSSD/snapshots/se-rssd_iter_104086.caffemodel \
--gpu all
# --weights=models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
