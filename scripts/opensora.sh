git clone -b opensora/v1.2 https://github.com/hpcaitech/Open-Sora.git

cp examples/oss_sora.py ./Open-Sora/opensora/schedulers/rf/__init__.py
cp examples/sample_oss_sora.py ./Open-Sora/opensora/utils/config_utils.py
cp examples/sora_inference_file.py ./Open-Sora/scripts/inference.py

export PYTHONPATH="./Open-Sora:$PYTHONPATH"

python3 Open-Sora/scripts/inference.py Open-Sora/configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --num-sampling-steps 200  --flow 5 --aes 6.5 \
  --prompt "a close-up shot of a woman standing in a dimly lit room. she is wearing a traditional chinese outfit, which includes a red and gold dress with intricate designs and a matching headpiece. the woman has her hair styled in an updo, adorned with a gold accessory. her makeup is done in a way that accentuates her features, with red lipstick and dark eyeshadow. she is looking directly at the camera with a neutral expression. the room has a rustic feel, with wooden beams and a stone wall visible in the background. the lighting in the room is soft and warm, creating a contrast with the woman's vibrant attire. there are no texts or other objects in the video. the style of the video is a portrait, focusing on the woman and her attire." --teacher-steps 200 --student-steps 20 --sample-name "OPEN-SORA_RESULT"
