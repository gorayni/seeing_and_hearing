#!/bin/bash

output_filepath="annotations/EPIC_subset_train_action_labels.csv"
train_labels_path='annotations/EPIC_train_action_labels.csv'

export valid_nouns=(1 3 4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 35 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 54 55 56 58 59 60 62 63 67 68 70 72 75 77 78 79 84 87 105 106 108 111 126)
export valid_verbs=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 32)

array_contains () { 
    local array="$1[@]"
    local seeking=$2
    local in=0
    for element in "${!array}"; do
        if [[ $element == $seeking ]]; then
            in=1
            break
        fi
    done
    echo $in
}

head -1 "$train_labels_path" > $output_filepath
tail -n +2 "$train_labels_path" | while read line 
do
  line=`echo "$line" | tr ' ' '#' | tr ',' ' '`
  verb_class=`echo $line | awk '{print $10}'`
  noun_class=`echo $line | awk '{print $12}'`

  is_valid_verb=`array_contains valid_verbs $verb_class`
  if [ "$is_valid_verb" -eq "0" ]; then
    continue
  fi
 
  is_valid_noun=`array_contains valid_nouns $noun_class`
  if [ "$is_valid_noun" -eq "0" ]; then
    continue
  fi

  echo $line >> $output_filepath
done

