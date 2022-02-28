for sid in $(seq 1 4); do
  for i in $(seq 0 11); do
    echo sbatch_regress.sh $sid $i
		sbatch sbatch_regress.sh $sid $i
	done
done
