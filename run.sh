TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nproc_per_node=8 --master_port 23456 \
main_no_decoder.py \
--cfg configs/small.yaml \
--data-path /public/liguoqi/imagenet1-k \
--batch-size 512 \
--tag ms_quant_tiny_no_decoder_ema \
--level-kind conv-conv \
--output /public/liguoqi/imagenet/spike_revcol_results
