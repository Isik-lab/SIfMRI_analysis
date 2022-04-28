#get files
subjs=( 1 2 3 4 )
#models=( "visual" "socialprimitive" "social" "indoor" "expanse" "agent_distance" "facingness" "transitivity" "joint_action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )
models=( "agent_distance" "joint_action" )

for subj in ${subjs[@]}; do
  for model in ${models[@]}; do
    sbatch sbatch_permutation.sh "$subj" "$model"
  done
done