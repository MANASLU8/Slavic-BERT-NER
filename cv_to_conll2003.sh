mkdir -p cv
for subset in rucv/*
do
	cp -r $subset cv
	IFS='/' # space is set as delimiter
	read -ra splitted_subset_name <<< "$subset" # str is read into an array as tokens separated by IFS
	IFS=' '
	subset_index=${splitted_subset_name[1]}
	cp rucv/$subset_index/test.txt cv/$subset_index/valid.txt
done

# set up structure
sed -i -E "s/(.+) (.+) (.+) (.+)/\1\t\4/" cv/*/*.txt
sed -i '1d' cv/*/*.txt
sed -i "1 i\<DOCSTART>" cv/*/*.txt

# set up tags
sed -i -E "s/I-Location|I-LocOrg/I-LOC/" cv/*/*.txt
sed -i -E "s/B-Location|B-LocOrg/B-LOC/" cv/*/*.txt
sed -i "s/I-Person/I-PER/" cv/*/*.txt
sed -i "s/B-Person/B-PER/" cv/*/*.txt
sed -i "s/I-Org/I-ORG/" cv/*/*.txt
sed -i "s/B-Org/B-ORG/" cv/*/*.txt