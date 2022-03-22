features=( "indoor" "expanse" "agent_distance" "facingness" "transitivity" "joint_action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )
controls=( "none" )#( "conv2" "conv5" "none" )
for control in "${controls[@]}"; do
  for feature in "${features[@]}"; do
    for sid in $(seq 1 4); do
      echo sbatch_permutation.sh "${sid}" "${feature}" "${control}"
      sbatch sbatch_permutation.sh "${sid}" "${feature}" "${control}"
    done
  done

  for feature in "${features[@]}"; do
    echo sbatch_permutation.sh "all" "${feature}" "${control}"
    sbatch sbatch_permutation.sh "all" "${feature}" "${control}"
  done

#  # Not by features
#  for sid in $(seq 1 4); do
#    echo sbatch_permutation.sh "${sid}" "not_by_feature" "${control}"
#    sbatch sbatch_permutation.sh "${sid}" "not_by_feature" "${control}"
#  done
#  echo sbatch_permutation.sh "all" "not_by_feature" "${control}"
#  sbatch sbatch_permutation.sh "all" "not_by_feature" "${control}"
done