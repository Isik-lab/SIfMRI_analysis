#get files
models=( "agent_distance" "arousal" "communication" "facingness" "joint_action"  "valence" )

for subj in 1 2 3 4; do
   for model in ${models[@]}; do
     sbatch sbatch_permutation.sh $subj $model
   done
done