#get files
subjs=( 1 2 3 4 )
models=( "all" "annotated" "nuissance" "visual" "socialprimitive" "social" )

for subj in ${subjs[@]}; do
  for model in ${models[@]}; do
      sbatch sbatch_permutation.sh "$subj" "$model"
  done
done
