features=( "indoor" "expanse" "agent_distance" "facingness" "transitivity" "joint_action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )
for feature in "${features[@]}"; do
  for sid in $(seq 1 4); do
    echo sbatch_permutation.sh "${sid}" "${feature}"
    sbatch sbatch_permutation.sh "${sid}" "${feature}"
  done
done

for feature in "${features[@]}"; do
  echo sbatch_permutation.sh "all" "${feature}"
  sbatch sbatch_permutation.sh "all" "${feature}"
done

for sid in $(seq 1 4); do
  echo sbatch_permutation.sh "${sid}" "not_by_feature"
  sbatch sbatch_permutation.sh "${sid}" "not_by_feature"
done

echo sbatch_permutation.sh "all" "not_by_feature"
#sbatch sbatch_permutation.sh "all" "not_by_feature"