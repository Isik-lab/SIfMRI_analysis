features=( 'indoor' 'expanse' 'agent distance' 'facingness' 'transitivity' 'joint action' 'communication' 'cooperation' 'dominance' 'intimacy' 'valence' 'arousal' )
for sid in $(seq 1 4); do
  for feature in "${features[@]}"; do
    echo sbatch_regress.sh "$sid" "$feature"
		sbatch sbatch_regress.sh "$sid" "$feature"
	done
done
