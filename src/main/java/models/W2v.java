package models;

import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import com.google.common.collect.ImmutableList;
import com.medallia.word2vec.Searcher;
import com.medallia.word2vec.Word2VecModel;
import com.medallia.word2vec.Word2VecTrainerBuilder;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.MultiKMeans;
import net.sf.javaml.clustering.evaluation.HybridCentroidSimilarity;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.distance.CosineDistance;
import net.sf.javaml.tools.Serial;
import pipes.FunctionToPipe;
import starter.Main;
import util.Utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by azaz on 07.08.17.
 */
public class W2v {
    public void W2VCluster() throws IOException, Searcher.UnknownWordException {
        System.out.println("reading model");
        Main.binModel = Word2VecModel.fromBinFile(new File("./models/w2v_02.bin"));
        Searcher search = Main.binModel.forSearch();
        Dataset data = new DefaultDataset();
        System.out.println("get W2v");
        for (String s : Main.binModel.getVocab()) {
//            System.out.println(s);
            ImmutableList<Double> rawVector = search.getRawVector(s);
            double[] arr = Utils.getDoubles(rawVector);
            data.add(new DenseInstance(arr));
        }
//        System.out.println(data.size());
        System.out.println("clustering");
        Clusterer clf = new MultiKMeans(50, 100, 10, new CosineDistance(), new HybridCentroidSimilarity());//new KMeans(50,100,new CosineDistance());
        Serial.store(clf, "./models/Clusterer_CLF.bin");
        Dataset[] clusters = clf.cluster(data);

        for (Dataset d : clusters) {
            for (int i = 0; i < Math.min(10, d.size()); i++) {
                System.out.println(search.getMatches(Utils.getDoubles(d.get(i).values()), 3));
            }
            System.out.println("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*==*=*=*");
        }

    }

    public void testW2VModel() throws IOException, Searcher.UnknownWordException {
        Main.binModel = Word2VecModel.fromBinFile(new File("./models/w2v_02.bin"));
        for (String s :
                "погашение_s задолженность_s заработный_a плата_s декабрь_s ндс_s облагаться_v фываыфва ".split(" ")
                ) {
            try {
                System.out.println(s + " " + Main.binModel.forSearch().getMatches(s, 10));
            } catch (Searcher.UnknownWordException e) {
            }
        }
        System.out.println(Main.binModel.forSearch().getRawVector("рублевый_a").size());
    }

    public  void w2vBuildModel() throws InterruptedException, IOException {
        Word2VecTrainerBuilder mod1 = Word2VecModel.trainer();
        mod1.setNumIterations(100);
        mod1.setWindowSize(10);                 //TODO tune
        mod1.useHierarchicalSoftmax();          //TODO tune
        mod1.type(NeuralNetworkType.SKIP_GRAM); //TODO tune
        mod1.setDownSamplingRate(1e-4);
        mod1.setMinVocabFrequency(2);
        mod1.setLayerSize(200);                  //TODO tune
        mod1.useNumThreads(4);

        ArrayList<List<String>> sentences = new ArrayList<List<String>>();
        ArrayList<Pipe> w2vPipeLine = new ArrayList<Pipe>();
        w2vPipeLine.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        w2vPipeLine.add(new TokenSequenceLowercase());

        w2vPipeLine.add(new FunctionToPipe((o) -> {
            sentences.add(((TokenSequence) o.getData()).stream().map(Token::getText).collect(Collectors.toList()));
            return o;
        }));

        InstanceList pipeline = new InstanceList(new SerialPipes(w2vPipeLine));
        pipeline.addThruPipe(
                new CsvIterator(
                        new FileReader(new File("stammed.txt")),
                        "(.*)", 1, -1, -1
                )
        );

        Word2VecModel model = mod1.train(sentences);
        System.out.println("Model builded");
        model.toBinFile(new FileOutputStream(new File("./models/w2v_02.bin")));
        System.out.println("Model saved");
    }
}

