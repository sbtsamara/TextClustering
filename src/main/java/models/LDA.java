package models;

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import com.medallia.word2vec.Searcher;
import pipes.Sentence2ArrayList;
import pipes.TokenSequence2File;
import pipes.TokenSequence2Stem;
import starter.Main;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class LDA {
    private ArrayList<Pipe> preprocessPipeList;
    private ArrayList<Pipe> testPipeList;
    private ArrayList<Pipe> trainPipeList;

    public LDA() {
        init();
    }

    public  void LDAOnText() throws Exception {

//        preprocessFile("filtered_logs_1.tsv", "stammed.txt");
//        ParallelTopicModel model = trainModel(50, "stammed.txt");
//        TestModel("models/model_Logs_2000.bin.2000");
        printModel("models/model_Logs_2000.bin.2000");
    }

    private void printModel(String filename) throws Exception {
        ParallelTopicModel model = ParallelTopicModel.read(new File(filename));
        System.out.println("Loaded");
        int i=0;
        for(Object[] words:model.getTopWords(10)){
            System.out.println("тема № "+i+" ключевые слова: "+Arrays.toString(words));


            i++;
        }
    }

    private void init() {
        preprocessPipeList = new ArrayList<Pipe>();
        preprocessPipeList.add(new CharSequenceLowercase());
        preprocessPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        preprocessPipeList.add(new TokenSequence2Stem());
        preprocessPipeList.add(new TokenSequenceLowercase());
        preprocessPipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));


        testPipeList = new ArrayList<Pipe>();
        testPipeList.add(new CharSequenceLowercase());
        testPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        testPipeList.add(new TokenSequence2Stem());
        testPipeList.add(new TokenSequenceLowercase());
        testPipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));
        testPipeList.add(new TokenSequence2FeatureSequence());

        trainPipeList = new ArrayList<Pipe>();
        trainPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        trainPipeList.add(new TokenSequenceLowercase());
        trainPipeList.add(new TokenSequence2FeatureSequence());

    }

    public  void preprocessFile(String input, String output) throws IOException {

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

    public  void TestModel(String filename) throws Exception {
        ParallelTopicModel model = ParallelTopicModel.read(new File(filename));
        System.out.println("Loaded");
        Object[][] topWords = model.getTopWords(10);
        ArrayList<String> arr = new ArrayList<String>();

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
                        new FileReader(new File("stem.txt")), "(.*)", 1, -1, -1
                )
        );

        TopicInferencer inferencer = model.getInferencer();
        ListIterator<String> it = arr.listIterator();
        for (Instance instance : instances) {
            String text = it.next();
            System.out.println(instance.getName() + " " + text);
            double[] sampledDistribution = inferencer.getSampledDistribution(instance, 50, 1, 5);
            TreeMap<Double, Integer> probs = new TreeMap<Double, Integer>();
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

    public  ParallelTopicModel trainModel(int topicCount, String filename) throws IOException {
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

    /*public List<Double> getVector(String s) {
        try {
            System.out.println(Main.binModel.forSearch().getMatches(s, 10));
            return Main.binModel.forSearch().getRawVector(s);

        } catch (Searcher.UnknownWordException e) {
            e.printStackTrace();
        }
        return null;


    }*/
}