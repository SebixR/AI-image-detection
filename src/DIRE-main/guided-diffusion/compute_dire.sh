## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
MODEL_PATH="../checkpoints/256x256_diffusion_uncond.pt" # "models/256x256_diffusion_uncond.pt"

SAMPLE_FLAGS="--batch_size 2 --num_samples 2 --timestep_respacing ddim20 --use_ddim True"

IMAGES_DIR_FLAG="--images_dir /home/user1/ml-project/data/train/resized/real"
RECONS_DIR_FLAG="--recons_dir /home/user1/ml-project/data/train/resized/reconstructions"
DIRE_DIR="--dire_dir /home/user1/ml-project/data/dire"

SAVE_FLAGS="$IMAGES_DIR_FLAG $RECONS_DIR_FLAG $DIRE_DIR"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=0 python compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS  $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder True