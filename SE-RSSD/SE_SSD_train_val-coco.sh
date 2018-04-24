#cd /home/soo/caffe_ssd
./build/tools/caffe train \
--solver=SE-RSSD/solver-coco.prototxt \
--weights=models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
--gpu all
# --weights=models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \