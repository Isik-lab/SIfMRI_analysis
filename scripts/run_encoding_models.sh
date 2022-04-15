subjs=( "1" "2" "3" "4" "all" )

# for subj in "${subjs[@]}"; do
#   #Full model
#   python voxel_encoding.py -s $subj --include_control --layer 2
#
#   #Predict by features
#   python voxel_encoding.py -s $subj --include_control --layer 2 --predict_by_feature
#
#   #Scene and object
#   python voxel_encoding.py -s $subj --include_control --layer 2 \
#   --predict_grouped_features -p "indoor" -p "expanse" -p "transitivity"
#
#   #Social primitives
#   python voxel_encoding.py -s $subj --include_control --layer 2 \
#   --predict_grouped_features -p "agent distance" -p "facingness"
#
#   #Social interaction
#   python voxel_encoding.py -s $subj --include_control --layer 2 \
#   --predict_grouped_features -p "joint action" -p "communication"
#
#   #Social
#   python voxel_encoding.py -s $subj --include_control --layer 2 \
#   --predict_grouped_features -p "cooperation" -p "dominance" -p "intimacy" \
#   -p "valence" -p "arousal"
#
#   #All social
#   python voxel_encoding.py -s $subj --include_control --layer 2 \
#   --predict_grouped_features -p "joint action" -p "communication" \
#   -p "cooperation" -p "dominance" -p "intimacy" \
#   -p "valence" -p "arousal"
# done

features=( "indoor" "expanse" "agent_distance" "facingness" "transitivity" "joint_action" "communication" "cooperation" "dominance" "intimacy" "valence" "arousal" )
for subj in "${subjs[@]}"; do
  for feature in "${features[@]}"; do
    python voxel_encoding.py -s $subj --include_control --layer 2 \
    --model_by_feature -m "$feature"
  done
done
