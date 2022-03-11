features=( "indoor" "expanse" "agent_distance" "facingness" "transitivity" "joint_action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )
for feature in "${feature[@]}"; done
  for sid in $(seq 1 4); do
    echo sbatch_permutation.sh "${sid}" by_feature
    sbatch sbatch_permutation.sh "${sid}" by_feature
  done
done

sbatch sbatch_permutation.sh all by_feature

for sid in $(seq 1 4); do
  echo sbatch_permutation.sh "${sid}"
  sbatch sbatch_permutation.sh "${sid}" not_by_feature
done

sbatch sbatch_permutation.sh all not_by_feature