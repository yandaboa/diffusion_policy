In your robodiff conda env (the one the diffusion_policy clone uses), make sure transformers is installed (pip install transformers — TransformerImagePolicy imports GPT2Config, GPT2Model). Then from /mnt/storage/lti/diffusion_policy/:

python eval_real_robot.py \
    -i <asteroid_ckpt> \
    -o tmp/rgb_eval_$(date +%s) \
    --robot_ip 192.168.1.10 -j