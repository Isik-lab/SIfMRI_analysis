#get files
FILES="../data/interim/VoxelEncoding/*y_pred.npy"
features=( "indoor" "expanse" "agent distance" "facingness" "transitivity" "joint action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )

for f in $FILES; do
  if [[ "$f" == *"predict-features"* ]]
  then
    for feature in "${features[@]}"; do
      sbatch sbatch_permutation.sh "$f" "$feature"
      echo sbatch_permutation.sh "$f" "$feature"
    done
  else
    sbatch sbatch_permutation.sh "$f"
    echo sbatch_permutation.sh "$f"
  fi
done