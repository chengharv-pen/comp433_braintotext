# COMP 433 Project Files

WORK IN PROGRESS

Clone the git repository
Assuming that we are at project root, I put the dataset in ./data and put the trained model in ./project_files/cnn_transformer/trained_models/cnn_transformer_attempt1
This also assumes that the command `nvidia-smi` works, if using a NVIDIA GPU

```
docker pull chengharvp/comp433-b2txt25-test

# Change shm-size if you have a lot more RAM. 12g signifies 12GB RAM.
docker run -it --rm --gpus all \
  --privileged \
  --shm-size=12g \
  -p 8888:8888 \
  -v ./:/workspace/comp433_braintotext25 \
  chengharvp/comp433-b2txt25-test:latest

cd comp433_braintotext25/language_model/runtime/server/x86
conda activate b2txt25_lm
python setup.py install

cd ../../../../
sysctl vm.overcommit_memory=1
redis-server --daemonize yes

# WARNING: THIS MAY OR MAY NOT WORK, DEPENDING ON YOUR RAM/VRAM
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0 &

cd project_files/cnn_transformer
conda activate b2txt25

# TO EVALUATE ON VALIDATION SET
python evaluate_model.py --model_path trained_models/cnn_transformer_attempt1 --data_dir ../../data/hdf5_data_final --eval_type val --gpu_number 0

# TO EVALUATE ON TEST SET (KAGGLE SUBMISSION)
python evaluate_model.py --model_path trained_models/cnn_transformer_attempt1 --data_dir ../../data/hdf5_data_final --eval_type test --gpu_number 0
```