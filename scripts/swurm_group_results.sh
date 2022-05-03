#get files
subjs=( 1 2 3 4 )
models=( "all" "annotated" "nuissance" "visual" "socialprimitive" "social" )

for model in ${models[@]}; do
    sbatch sbatch_group_results.sh "$model"
done
