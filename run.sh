TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nproc_per_node=8 --master_port 23456 \
main_no_decoder.py \
--cfg configs/small.yaml \
--data-path /public/liguoqi/imagenet1-k \
--batch-size 512 \
--tag small_baseline \
--output /public/liguoqi/imagenet/t_revsnn
