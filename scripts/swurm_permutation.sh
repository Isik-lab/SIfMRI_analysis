#get files
models=( "agent_distance" "facingness" "joint_action" "communication" "valence" "arousal" )

for subj in 1 2 3 4; do
   for model in ${models[@]}; do
     sbatch sbatch_permutation.sh $subj $model
   done
done