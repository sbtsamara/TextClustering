
/**
 * Created by azaz on 25.07.17.
 */

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import com.google.common.collect.ImmutableList;
import com.medallia.word2vec.Searcher;
import com.medallia.word2vec.Word2VecModel;
import com.medallia.word2vec.Word2VecTrainerBuilder;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import net.sf.javaml.clustering.*;
import net.sf.javaml.clustering.evaluation.HybridCentroidSimilarity;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.distance.CosineDistance;
import net.sf.javaml.tools.weka.WekaClusterer;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Main {

    public static final File binFile = new File("/home/azaz/PycharmProjects/SBT/Models/ruscorpora_mean_hs.model.bin");
    public static Word2VecModel binModel;
    private static ArrayList<Pipe> preprocessPipeList;
    private static ArrayList<Pipe> testPipeList;
    private static ArrayList<Pipe> trainPipeList;


    public static void main(String[] args) throws Exception {
//        w2vBuildModel();

//        testW2VModel();
        System.out.println("reading model");
        binModel = Word2VecModel.fromBinFile(new File("./models/w2v_02.bin"));
        Searcher search = binModel.forSearch();
        Dataset data = new DefaultDataset();
        System.out.println("get W2v");
        for (String s : binModel.getVocab()) {
//            System.out.println(s);
            ImmutableList<Double> rawVector = search.getRawVector(s);
            double[] arr = getDoubles(rawVector);
            data.add(new DenseInstance(arr));
        }
//        System.out.println(data.size());
        System.out.println("clustering");
        Clusterer clf = new MultiKMeans(50,100,10,new CosineDistance(),new HybridCentroidSimilarity());//new KMeans(50,100,new CosineDistance());
        Dataset[] clusters=clf.cluster(data);
        for(Dataset d:clusters){
            for (int i = 0; i < Math.min(10,d.size()); i++) {
                System.out.println(search.getMatches(getDoubles(d.get(i).values()),3));
            }
            System.out.println("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*==*=*=*");
        }
//        model.getVocab().forEach(System.out::println);
//        LDAOnText();
    }

    private static double[] getDoubles(Collection<Double> rawVector) {
        double[] arr= new double[rawVector.size()];
        int i=0;
        for(Double d:rawVector){
            arr[i++]=d;
        }
        return arr;
    }

    private static void testW2VModel() throws IOException {
        binModel = Word2VecModel.fromBinFile(new File("./models/w2v_02.bin"));
        for (String s :
                "погашение_s задолженность_s заработный_a плата_s декабрь_s ндс_s облагаться_v фываыфва ".split(" ")
                ) {
            try {
                System.out.println(s + " " + binModel.forSearch().getMatches(s, 10));
            } catch (Searcher.UnknownWordException e) {
            }
//            System.out.println(binModel.forSearch rch().getRawVector("рублевый_a"));
        }
    }

    private static void w2vBuildModel() throws InterruptedException, IOException {
        Word2VecTrainerBuilder mod1 = Word2VecModel.trainer();
        mod1.setNumIterations(100);
        mod1.setWindowSize(10);                 //TODO tune
        mod1.useHierarchicalSoftmax();          //TODO tune
        mod1.type(NeuralNetworkType.SKIP_GRAM); //TODO tune
        mod1.setDownSamplingRate(1e-4);
        mod1.setMinVocabFrequency(2);
        mod1.setLayerSize(200);                  //TODO tune
        mod1.useNumThreads(4);

        ArrayList<List<String>> sentences = new ArrayList<>();
        ArrayList<Pipe> w2vPipeLine = new ArrayList<>();
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

    private static void LDAOnText() throws Exception {
        init();
        preprocessFile("filtered_logs_1.tsv", "stammed.txt");
        ParallelTopicModel model = trainModel(50, "stammed.txt");
        TestModel("models/model_Logs_2000.bin.2000");
    }

    private static void init() {
        preprocessPipeList = new ArrayList<>();
        preprocessPipeList.add(new CharSequenceLowercase());
        preprocessPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        preprocessPipeList.add(new TokenSequence2Stem());
        preprocessPipeList.add(new TokenSequenceLowercase());
        preprocessPipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));


        testPipeList = new ArrayList<>();
        testPipeList.add(new CharSequenceLowercase());
        testPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        testPipeList.add(new TokenSequence2Stem());
        testPipeList.add(new TokenSequenceLowercase());
        testPipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));
        testPipeList.add(new TokenSequence2FeatureSequence());

        trainPipeList = new ArrayList<>();
        trainPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        trainPipeList.add(new TokenSequenceLowercase());
        trainPipeList.add(new TokenSequence2FeatureSequence());

    }

    private static void preprocessFile(String input, String output) throws IOException {

        PrintWriter pw = new PrintWriter(new FileWriter(new File(output)), false);
        preprocessPipeList.add(new TokenSequence2File(pw));

        InstanceList instances = new InstanceList(new SerialPipes(preprocessPipeList));


        Pattern p = Pattern.compile("" +
                "([^\t]*\\t){6}" +
                "([^\t]*\\t)" +
                "(.*)");
        instances.addThruPipe(
                new CsvIterator(
                        new FileReader(new File(input)),
                        p, 2, -1, -12
                )
        );
        pw.flush();
        pw.close();

    }

    private static void TestModel(String filename) throws Exception {
        ParallelTopicModel model = ParallelTopicModel.read(new File(filename));
        System.out.println("Loaded");
        Object[][] topWords = model.getTopWords(10);
        ArrayList<String> arr = new ArrayList<>();

        testPipeList.add(1, new Sentence2ArrayList(arr));
        InstanceList instances = new InstanceList(new SerialPipes(testPipeList));


        /*instances.addThruPipe(
                new StringArrayIterator(
                        new String[]{
                                "при сохранении документа это указывается как обязательное поле",
                                "С данным заявление только в банк по месту обслуживания. Служба технической поддержки не принимает и не обрабатывает электронные документы,письма и платежные поручения."
                        }

                )
        );*/
        instances.addThruPipe(
                new CsvIterator(
                        new FileReader(new File("qwe.txt")), "(.*)", 1, -1, -1
                )
        );

        TopicInferencer inferencer = model.getInferencer();
        ListIterator<String> it = arr.listIterator();
        for (Instance instance : instances) {
            String text = it.next();
            System.out.println(instance.getName() + " " + text);
            double[] sampledDistribution = inferencer.getSampledDistribution(instance, 50, 1, 5);
            TreeMap<Double, Integer> probs = new TreeMap<>();
            int[] k = {0};

            Arrays.stream(sampledDistribution).forEach(v -> probs.put(v, k[0]++));
            double p0 = -1;
            for (Map.Entry<Double, Integer> e : probs.descendingMap().entrySet()) {
                if (p0 < 0) {
                    p0 = e.getKey();
                    if (p0 < 0.1) {
                        break;
                    }
                } else if (e.getKey() < p0 / 20) {
                    break;
                }
                System.out.println(e.getKey() + "\t" + e.getValue() + "\t" + Arrays.toString(topWords[e.getValue()]));
            }
            System.out.println("============================================");
        }
    }

    public static ParallelTopicModel trainModel(int topicCount, String filename) throws IOException {
        ParallelTopicModel model = new ParallelTopicModel(topicCount);

        InstanceList pipeline = new InstanceList(new SerialPipes(trainPipeList));

        Pattern p = Pattern.compile("(.*)");
        pipeline.addThruPipe(
                new CsvIterator(
                        new FileReader(new File(filename)),
                        p, 1, -1, -1
                )
        );

        model.addInstances(pipeline);

        model.setNumThreads(4);
        model.setNumIterations(2000);
        model.setSaveSerializedModel(500, "./models/model_Logs_2000.bin");
        model.estimate();

        return model;
    }

    private static List<Double> getVector(String s) {
        try {
            System.out.println(binModel.forSearch().getMatches(s, 10));
            return binModel.forSearch().getRawVector(s);

        } catch (Searcher.UnknownWordException e) {
            e.printStackTrace();
        }
        return null;


    }
}
