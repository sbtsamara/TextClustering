package ru.azaz.textProcessing.models;

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import ru.azaz.textProcessing.pipes.Sentence2Collection;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.regex.Pattern;

public class StopWords {
    public List<String> stopWords;
    public void generate(String file) throws FileNotFoundException {

        ArrayList<String> texts = new ArrayList<String>();
        HashSet<String> words = new HashSet<>();
        ArrayList<Pipe> list = new ArrayList<>();

        list.add(new CharSequenceLowercase());
        list.add(new Sentence2Collection(texts));
        list.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        list.add(new Sentence2Collection(words));
        InstanceList instances = new InstanceList(new SerialPipes(list));
        instances.addThruPipe(
                new CsvIterator(
                        new FileReader(new File(file)), "(.*)", 1, -1, -1
                )
        );

        TfidfVectorizer vec = new TfidfVectorizer.Builder().setIterator(new CollectionSentenceIterator(texts)).setTokenizerFactory(new DefaultTokenizerFactory()).build();
        vec.fit();


        ArrayList<String> result = new ArrayList<>();

        HashMap<String,Double> score =new HashMap<>();
        double max=-1;
        for(String s:words) {
            double tfidfWord = vec.tfidfWord(s, 1, 1);
//            System.out.println(s+" "+tfidfWord);
            max=Double.max(max,Double.isNaN(tfidfWord)?-1:tfidfWord);
            score.put(s, tfidfWord);
            if(score.get(s).isNaN()){
                result.add(s);
            }
        }

        double split=0.6;//10% quantile
        System.out.println(max);
        for(String s:words) {
//            System.out.println(s+" "+(score.get(s)/max));
//            System.out.println(score.get(s));
            if((score.get(s)/max) <split){
                result.add(s);
            }
        }

//        System.out.println(result);
        stopWords=result;
    }
    public void writeToFile(String toFile) throws FileNotFoundException {
        PrintWriter pw = new PrintWriter(new FileOutputStream(new File(toFile),false));
        stopWords.forEach(sw->pw.print(sw+" "));
        pw.flush();
        pw.close();
    }
}
