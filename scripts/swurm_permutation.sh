#get files
subjs=( 1 2 3 4 )
models=( "all" "visual" "primitive" "social" "affective" "nuissance" "lowhighvis" "annotated" "indoor" "expanse" "transitivity" "agent distance" "facingness" "joint action" "communication" "valence" "arousal" )



for subj in ${subjs[@]}; do
  for model in ${models[@]}; do
      sbatch sbatch_permutation.sh "$subj" "$model"
  done
done
