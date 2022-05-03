#get files
models=( "all" "annotated" "visual" "socialprimitive" "social" )

for model in ${models[@]}; do
    sbatch sbatch_group_results.sh "$model"
done
