#features=( "indoor" "expanse" "agent_distance" "facingness" "transitivity" "joint_action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )
for sid in $(seq 1 4); do
  echo sbatch_permutation.sh "${sid}"
  sbatch sbatch_permutation.sh "${sid}"
done
