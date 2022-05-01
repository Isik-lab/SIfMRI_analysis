#get files
subjs=( 1 2 3 4 )
models=( "all" "visual" "socialprimitive" "social" "indoor" "expanse" "agent_distance" "facingness" "transitivity" "joint_action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )
cvs=( "" "--cross_validation")

for subj in ${subjs[@]}; do
  for model in ${models[@]}; do
    for cv in ${cvs[@]}; do
      sbatch sbatch_permutation.sh "$subj" "$model" "$cv"
    done
  done
done
