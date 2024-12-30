PROBS=0.9
RAW_RESULT=$(echo "1 - $PROBS" | bc -l)
RESULT=$(printf "%.9g\n" "$RAW_RESULT")
echo $RESULT
