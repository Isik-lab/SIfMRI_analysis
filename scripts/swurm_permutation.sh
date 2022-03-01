features=( "indoor" "expanse" "agent_distance" "facingness" "transitivity" "joint_action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )
for sid in $(seq 1 4); do
  for feature in "${features[@]}"; do
    echo sbatch_permutation.sh "${sid}" "${feature}"
		sbatch sbatch_permutation.sh "${sid}" "${feature}"
	done
done
