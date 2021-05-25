status_echo () {
n_files=`ls /storage/projects/instpred/results/evaluate/1_evaluate_f2f_nT3/t+3/*.txt | wc -l`
percent_files=$((100*$n_files/15000))
end_time="$(date -u +%s)"
elapsed="$(($end_time - $start_time))"
elapsed_files=$(($n_files - $n_files_start))
if [ "$elapsed_files" -eq "0" ]; then elapsed_files=1; fi
remaining_files=$((15000-$n_files))
remaining=$(($elapsed * $remaining_files / $elapsed_files))
echo $n_files $percent_files'%' elapsed:$(($elapsed / 3600))h$((($elapsed / 60)%60))m$(($elapsed % 60))s remaining:$(($remaining / 3600))h$((($remaining / 60)%60))m$(($remaining % 60))s
}
export -f status_echo

export n_files_start=`ls /storage/projects/instpred/results/evaluate/1_evaluate_f2f_nT3/t+3/*.txt | wc -l`
export start_time="$(date -u +%s)"
watch -n 10 -x bash -c status_echo