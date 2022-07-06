subjs=( 1 2 3 4 )
models=( "facingness" "joint_action" "communication" )

for subj in ${subjs[@]}; do
  python voxel_regression.py -s $subj --cross_validation
  python voxel_permutation.py -s $subj --cross_validation --n_perm 1
done

for subj in ${subjs[@]}; do
  for model in ${models[@]}; do
      python voxel_regression.py -s $subj --cross_validation --unique_model $model
      python voxel_permutation.py -s $subj --cross_validation --n_perm 1 --unique_model $model
  done
done
