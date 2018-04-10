echo "ssplit.eolonly=true" > line.properties
CORENLP_ROOT="../stanford-corenlp-full-2017-06-09"
for dir_name in train test dev
do
    for t in a.txt b.txt
    do
        java -cp "$CORENLP_ROOT/*" \
            -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP \
            -annotators tokenize,ssplit,pos,lemma,parse \
            -file "data/"${dir_name}"/"${t} \
            -outputFormat json \
            -outputDirectory "data/"${dir_name} \
            -props line.properties
    done
done
rm line.properties
