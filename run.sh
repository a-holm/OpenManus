while IFS= read -r line; do python3 script.py "$line"; done < prompt.txt
