git clone https://github.com/Wan-Video/Wan2.1.git

cp examples/oss_wan.py Wan2.1/wan/text2video.py
cp examples/sample_oss_wan.py Wan2.1/generate.py

CKPT_PATH=./Wan2.1-T2V-14B

python Wan2.1/generate.py  --task t2v-14B --size 832*480 --ckpt_dir $CKPT_PATH  --prompt "一名男子在跳台上做专业跳水动作。全景平拍镜头中，他穿着红色泳裤，身体呈倒立状态，双臂伸展，双腿并拢。镜头下移，他跳入水中，溅起水花。背景中是蓝色的泳池。" --base_seed 4 --sample_steps 200 --sample_solver dpm++ --solver_order 1 --frame_num 81 --save_file "wan_result.mp4" --student_steps 20  --norm 2 --frame_type "4" --channel_type "all"
