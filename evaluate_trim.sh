export N_FUTURE_FRAMES=6

# Val
python eval.py --model models/F2F/F2F.net -nTR $N_FUTURE_FRAMES --nb_scales 1,1,1,3 --id_gpu_model 0 --save evaluate/1_evaluate_f2f_nT3_trim --nvalIt 2

# Test
python eval.py --model models/F2F/F2F.net -nTR $N_FUTURE_FRAMES --nb_scales 1,1,1,3 --id_gpu_model 0 --save evaluate/2_evaluate_f2f_nT3_test_set_trim --test_set True --nvalIt 2

# for i in $(seq 1 $N_FUTURE_FRAMES)
# do
#     export PREDICTIONS_PATH=evaluate/2_evaluate_f2f_nT3_test_set_trim/t+${i}
#     export EVALUATION_RESULTS=evaluate/1_evaluate_f2f_nT3_results_t+${i}_trim
#     python eval_precomputed_results.py
#     mkdir results/evaluate/final
#     mkdir results/evaluate/final/pred_t+${i}/
#     cp results/evaluate/1_evaluate_f2f_nT3_results_t+${i}_trim/outputs_semantic_segmentation/* results/evaluate/final/${i}/
# done

export PREDICTIONS_PATH=evaluate/1_evaluate_f2f_nT3/t+2
export EVALUATION_RESULTS=evaluate/1_evaluate_f2f_nT3_results_t+2
python eval_precomputed_results.py
