#get files
subjs=( 1 2 3 4 )
models=( "visual" "socialprimitive" "social" "indoor" "expanse" "agent distance" "facingness" "transitivity" "joint action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )

for subj in ${subjs[@]}; do
  for model in ${models[@]}; do
    sbatch sbatch_permutation.sh "$subj" "$model"
  done
done