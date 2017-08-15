package ru.azaz.textProcessing.models;

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.pipe.iterator.StringArrayIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import javafx.util.Pair;
import ru.azaz.textProcessing.pipes.Sentence2Collection;
import ru.azaz.textProcessing.pipes.TokenSequence2File;
import ru.azaz.textProcessing.pipes.TokenSequence2Stem;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import static ru.azaz.textProcessing.util.Utils.pairComparator;

public class LDA {
    private ArrayList<Pipe> preprocessPipeList;
  //  private ArrayList<Pipe> testPipeList;
//    private ArrayList<Pipe> trainPipeList;

    public LDA() {
        init();
    }

    public void LDAOnText() throws Exception {

//        preprocessFile("filtered_logs_1.tsv", "stammed.txt");
//        ParallelTopicModel model = trainModel(50, "stammed.txt");
//        TestModel("models/model_Logs_2000.bin.2000");
        printModel("models/model_Logs_2000.bin.2000");
    }

    public void printModel(String filename) throws Exception {
        ParallelTopicModel model = ParallelTopicModel.read(new File(filename));
        System.out.println("Loaded");
        printModel(model);
    }

    public void printModel(ParallelTopicModel model) throws Exception {
        int i = 0;
        for (Object[] words : model.getTopWords(10)) {
            System.out.println("тема №" + i + "ключевые слова:" + Arrays.toString(words));
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


        /*testPipeList = new ArrayList<Pipe>();
        testPipeList.add(new CharSequenceLowercase());
        testPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        testPipeList.add(new TokenSequence2Stem());
        testPipeList.add(new TokenSequenceLowercase());
        testPipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));
        testPipeList.add(new TokenSequence2FeatureSequence());
*/

    }

    public void preprocessFile(String input, String output) throws IOException {

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

    public void TestModel(String modelPath, String onFile) throws Exception {
        ParallelTopicModel model = ParallelTopicModel.read(new File(modelPath));
        System.out.println("Loaded");
        TestModel(model, onFile);
    }

    public void TestModel(ParallelTopicModel model, String onFile) throws Exception {

        Object[][] topWords = model.getTopWords(10);

        ArrayList<String> arr = new ArrayList<String>();
        ArrayList<Pipe> list = new ArrayList<>();

        list.add(new CharSequenceLowercase());
        list.add(new Sentence2Collection(arr));
        list.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        list.add(new TokenSequence2FeatureSequence(model.getAlphabet()));

        InstanceList instances = new InstanceList(new SerialPipes(list));

        instances.addThruPipe(
                new CsvIterator(
                        new FileReader(new File(onFile)), "(.*)\\*=\\*=\\*=(.*)", 2, 1, -1
                )
        );
        Iterator<Instance> instanceIterator = instances.iterator();

        TopicInferencer inferencer = model.getInferencer();
        model.setRandomSeed(1337);
        inferencer.setRandomSeed(1337);

        int[] counter = new int[model.getNumTopics()];
        ArrayList<Pair<Double, String>>[] examples = new ArrayList[model.getNumTopics()];
        for (int i = 0; i < examples.length; i++) {
            examples[i] = new ArrayList<>();
        }

        ListIterator<String> it = arr.listIterator();
        while (instanceIterator.hasNext()) {
            Instance inst = instanceIterator.next();
            double[] results = inferencer.getSampledDistribution(inst, 0, 0, 0);
            int maxInd = 0;
            for (int j = 0; j < results.length; j++) {
                if (results[maxInd] < results[j]) {
                    maxInd = j;
                }
            }
            counter[maxInd]++;
            examples[maxInd].add(new Pair<>(results[maxInd], "\"" + inst.getTarget() + "\";\"" + it.next() + "\""));
        }

        for (int i = 0; i < examples.length; i++) {
            examples[i].sort(pairComparator.reversed());
        }

        Object[][] topWord = model.getTopWords(10);
        for (int i = 0; i < counter.length; i++) {
            System.out.println("тема №" + i + "; ключевые слова:" + Arrays.toString(topWord[i]) + "; количество документов по теме: " + counter[i]);
        }

        System.out.println("theme;text;stammed;keywords");
        for (int i = 0; i < counter.length; i++) {
            for (Pair<Double, String> example : examples[i]) {
                System.out.println(i + ";"+ example.getValue() + ";" + Arrays.toString(topWord[i]));
            }
        }

    }

    private void printWords(Object[][] topWords, InstanceList instances, TopicInferencer inferencer, ListIterator<String> it) {
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

    public void evaluateMode(ParallelTopicModel model, String texts) {

        ArrayList<String> arr = new ArrayList<String>();
        ArrayList<Pipe> list = new ArrayList<>();
        list.add(new CharSequenceLowercase());
//        list.remove(list.size()-1);
//        list.remove(list.size()-1);
        list.add(new Sentence2Collection(arr));
        list.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        list.add(new TokenSequence2FeatureSequence(model.getAlphabet()));

        Object[][] topWords = model.getTopWords(10);
        InstanceList instances = new InstanceList(new SerialPipes(list));

        instances.addThruPipe(
                new StringArrayIterator(
                        new String[]{texts}
                )
        );
        TopicInferencer inferencer = model.getInferencer();
        ListIterator<String> it = arr.listIterator();
        printWords(topWords, instances, inferencer, it);

    }

    public ParallelTopicModel trainModel(int topicCount, String filename) throws IOException {
        return trainModel(topicCount, filename, null);
    }

    public ParallelTopicModel trainModel(int topicCount, String filename, String output) throws IOException {
        return trainModel(topicCount, 500, filename, output);
    }

    public ParallelTopicModel trainModel(int topicCount, int iterations, String filename, String output) throws IOException {
        ParallelTopicModel model = new ParallelTopicModel(topicCount);
        return trainModel(model, topicCount, iterations, filename, output);
    }

    public ParallelTopicModel trainModel(ParallelTopicModel model, int topicCount, int iterations, String filename, String output) throws IOException {
        return trainModel(model, topicCount, iterations, filename, output, "(.*)", 1);
    }

    public ParallelTopicModel trainModel(ParallelTopicModel model, int topicCount, int iterations, String filename, String output, String regexp, int dataGroup) throws IOException {

        ArrayList<Pipe> trainPipeList = new ArrayList<Pipe>();
        trainPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        trainPipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));
        trainPipeList.add(new TokenSequenceLowercase());
        trainPipeList.add(new TokenSequence2FeatureSequence(model.getAlphabet()==null?new Alphabet():model.getAlphabet()));
        InstanceList pipeline = new InstanceList(new SerialPipes(trainPipeList));

        Pattern p = Pattern.compile(regexp);
        pipeline.addThruPipe(
                new CsvIterator(
                        new FileReader(new File(filename)),
                        p, dataGroup, -1, -1
                )
        );

        model.addInstances(pipeline);
        model.setNumIterations(iterations);
        model.setRandomSeed(1337);
        model.estimate();

        File f = new File(output + "/");
        f.mkdirs();
        model.write(new File(output + "/model_Topic:" + topicCount + "_IT:" + iterations + ".bin"));
//        pipeline.getAlphabet().
        return model;
    }


}