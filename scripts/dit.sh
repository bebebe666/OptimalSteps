git clone https://github.com/facebookresearch/DiT.git

cp examples/sample_oss_dit.py ./DiT

CKPT_PATH=DiT_CKPT_PATH
FOLDER_NAME=OUTPUT_DIR

torchrun --nproc_per_node=1 --master_port=12345 --nnodes=1 DiT/sample_oss_dit.py  --model DiT-XL/2 --ckpt $CKPT_PATH --cfg-scale 1.5  --folder_name $FOLDER_NAME --num-classes 1000 --search_batch 1 --student_step 5 --teacher_step 200 --renorm_flag --search_each
