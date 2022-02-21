for sid in $(seq 1 4); do
  for i in $(seq 0 11); do
		sbatch sbatch_regress.sh $sid $i
	done
done
