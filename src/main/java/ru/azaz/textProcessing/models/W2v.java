package ru.azaz.textProcessing.models;

import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.MultiKMeans;
import net.sf.javaml.clustering.evaluation.HybridCentroidSimilarity;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.distance.CosineDistance;
import net.sf.javaml.tools.Serial;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import ru.azaz.textProcessing.pipes.FunctionToPipe;
import ru.azaz.textProcessing.util.Utils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Pattern;

/**
 * Created by azaz on 07.08.17.
 */
public class W2v {
//    public static Word2VecModel binModel;
    public static Word2Vec model;

    public void W2VCluster() throws IOException{
        System.out.println("reading model");

        model = WordVectorSerializer.readWord2VecModel(new File("./models/w2v_02.bin"));
//        Searcher search = binModel.forSearch();
        Dataset data = new DefaultDataset();
        System.out.println("get W2v");
        for (String s : model.getVocab().words()) {
//            System.out.println(s);
            double[] rawVector = model.getWordVector(s);
            data.add(new DenseInstance(rawVector));
        }
//        System.out.println(data.size());
        System.out.println("clustering");
        Clusterer clf = new MultiKMeans(50, 100, 10, new CosineDistance(), new HybridCentroidSimilarity());//new KMeans(50,100,new CosineDistance());
        Serial.store(clf, "./models/Clusterer_CLF.bin");
        Dataset[] clusters = clf.cluster(data);

        for (Dataset d : clusters) {
            for (int i = 0; i < Math.min(10, d.size()); i++) {
                double [][] arr1=new double[1][d.get(i).values().size()];
                arr1[0]=Utils.getDoubles(d.get(i).values());
                System.out.println(model.wordsNearest(new NDArray(arr1), 3));
            }
            System.out.println("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*==*=*=*");
        }

    }

    public void testW2VModel() throws IOException {
//        binModel = Word2VecModel.fromBinFile(new File("./models/w2v_02.bin"));
        model = WordVectorSerializer.readWord2VecModel(new File("./models/w2v_02.bin"));
        for (String s :
                "погашение_s задолженность_s заработный_a плата_s декабрь_s ндс_s облагаться_v фываыфва ".split(" ")
                ) {
            System.out.println(s + " " + model.wordsNearest(s, 10));

        }
        System.out.println(model.getWordVector("рублевый_a").length);
    }

    public void w2vBuildModel() throws InterruptedException, IOException {
        ArrayList<String> sentences = new ArrayList<String>();
        ArrayList<Pipe> w2vPipeLine = new ArrayList<Pipe>();
        w2vPipeLine.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        w2vPipeLine.add(new TokenSequenceLowercase());

        w2vPipeLine.add(new FunctionToPipe((o) -> {
            sentences.add(((TokenSequence) o.getData()).stream().map(Token::getText).reduce((s, s2) -> s+" "+s2).get());
            return o;
        }));

        InstanceList pipeline = new InstanceList(new SerialPipes(w2vPipeLine));
        pipeline.addThruPipe(
                new CsvIterator(
                        new FileReader(new File("stammed.txt")),
                        "(.*)", 1, -1, -1
                )
        );

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(2)
                .iterations(100)
                .layerSize(200)
                .useHierarchicSoftmax(true)
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .workers(4)
                .seed(42)
                .windowSize(10)
                .iterate(new CollectionSentenceIterator(sentences))
                .tokenizerFactory(new DefaultTokenizerFactory())
                .build();
        vec.fit();
        System.out.println("Model builded");
        WordVectorSerializer.writeWord2VecModel(vec,new File("./models/w2v_02.bin"));
        System.out.println("Model saved");
    }
}

