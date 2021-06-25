# F2F5
#------
# -trit : Number of training iterations per epoch 'ntrainIt'
# -trbs : Number of samples per batch for training iterations 'trainbatchsize'
# -nEps : Save every nEps epochs 'nEpocheSave'
# -nEp  : Train for nEp epochs 'nEpoches'
# -tsl  : Feature name for single level training 'train_single_level'
# --nb_scales : Number of scales in the the model
# -nTR  : Number of target frames 'n_target_frames_ar'
# -----
# python train.py -lr 0.01 -trit 743 -trbs 4 --save train/1_f2f5_nT1 -nEps 10 -tsl fpn_res5_2_sum  --nb_scales 1 -nTR 1 -nEp 80
python train.py -lr 0.01 -trit 1 -tsl fpn_res5_2_sum --nb_scales 1 -nTR 1 -nEp 1


#------
# F2F4
#------
# --model : Model to initialize weights with
#------
# python train.py --nb_scales 1 -lr 0.005 -nEp 80 -tsl fpn_res4_5_sum --id_gpu_model 1 -nTR 1 --save train/2_ft_f2f5_for_p4_nT1 --model results/train/1_f2f5_nT1/model_80ep.net
# python train.py --nb_scales 1 -lr 0.005 -trit 10 -nEp 1 -tsl fpn_res4_5_sum --id_gpu_model 0 -nTR 1 --save train/2_ft_f2f5_for_p4_nT1 --model results/train/1_f2f5_nT1/model_80ep.net
