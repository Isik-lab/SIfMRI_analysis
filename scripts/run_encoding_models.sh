subjs=( "1" "2" "3" "4" "all" )
features=( "indoor" "expanse" "agent distance" "facingness" "transitivity" "joint action" "communication" "valence" "arousal" )

for subj in "${subjs[@]}"; do
  #Full model
  python voxel_encoding.py -s $subj --include_control --layer 2

  #Predict by features
  python voxel_encoding.py -s $subj --include_control --layer 2 --predict_by_feature

  #Scene and object
  python voxel_encoding.py -s $subj --include_control --layer 2 \
  --predict_grouped_features -p "indoor" -p "expanse" -p "transitivity"

  #Social primitives
  python voxel_encoding.py -s $subj --include_control --layer 2 \
  --predict_grouped_features -p "agent distance" -p "facingness"

  #Social interaction
  python voxel_encoding.py -s $subj --include_control --layer 2 \
  --predict_grouped_features -p "joint action" -p "communication"

  #Social
  python voxel_encoding.py -s $subj --include_control --layer 2 \
  --predict_grouped_features -p "cooperation" -p "dominance" -p "intimacy" \
  -p "valence" -p "arousal"

  #All social
  python voxel_encoding.py -s $subj --include_control --layer 2 \
  --predict_grouped_features -p "joint action" -p "communication" \
  -p "cooperation" -p "dominance" -p "intimacy" \
  -p "valence" -p "arousal"

  for feature in "${features[@]}"; do
    python voxel_encoding.py -s $subj --include_control --layer 2 \
    --model_by_feature -m "$feature"
  done
done
