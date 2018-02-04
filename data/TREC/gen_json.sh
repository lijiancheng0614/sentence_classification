echo "ssplit.eolonly=true" > line.properties
CORENLP_ROOT="../stanford-corenlp-full-2017-06-09"
for filename in sentences_train.txt sentences_test.txt
do
    java -cp "$CORENLP_ROOT/*" \
        -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP \
        -annotators tokenize,ssplit,pos,lemma,parse \
        -file $filename \
        -outputFormat json \
        -outputDirectory . \
        -props line.properties
done
rm line.properties
