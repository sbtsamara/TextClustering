package starter;

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import org.deeplearning4j.datasets.DataSets;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import pipes.FunctionToPipe;
import pipes.TokenSequence2Stem;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.regex.Pattern;

/**
 * Created by azaz on 07.08.17.
 */
public class doc2vecTest {

    public static boolean TRAIN = false;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
//        TRAIN=true;

//        preprocessFile("./data/typed.csv","./data/stammed_clf.csv",true);
//        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("./data/stammed_clf.csv")));
        ParagraphVectors build = null;
        if (TRAIN) {
            build = trainDoc2Vec();
        } else {
            build = WordVectorSerializer.readParagraphVectors("models/Doc2vec_CUDA.bin");
            build.setTokenizerFactory(new DefaultTokenizerFactory());
        }

//        testDoc2vec(build);
        NN(build);

    }

    private static void NN(ParagraphVectors d2v) throws IOException {
        LabelAwareSentenceIterator train = new LabelAwareListSentenceIterator(new FileInputStream("./data/train_clf.csv"), ":::::::", 1, 0);
        LabelAwareSentenceIterator test = new LabelAwareListSentenceIterator(new FileInputStream("./data/test_clf.csv"), ":::::::", 1, 0);

        HashMap<String, Integer> kv = new HashMap<>();
        int i = 0;
        int len = 0;
        while (train.hasNext()) {
            len++;
            train.nextSentence();
            if (!kv.containsKey(train.currentLabel())) {
                kv.put(train.currentLabel(), i++);
            }
        }
        System.out.println(kv);
        train.reset();

//        MultiLayerNetwork model = trainNN(d2v, train, kv);
        MultiLayerNetwork model=ModelSerializer.restoreMultiLayerNetwork("./models/NN.bin");
        System.out.println("loaded");
        testNN(d2v, test, kv, model);
//        model.fit(new NDArray(sentences),new NDArray(oneHot));
//        model.fit();
    }

    @NotNull
    private static MultiLayerNetwork trainNN(ParagraphVectors d2v, LabelAwareSentenceIterator train, HashMap<String, Integer> kv) throws IOException {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1337)
                .miniBatch(true)
                .iterations(350)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .updater(Updater.SGD)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(d2v.getLayerSize()).nOut(150)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(150).nOut(kv.size()).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

        ArrayList<double[]> oneHot = new ArrayList<>();// new double[len][kv.size()];
        ArrayList<double[]> sentences=new ArrayList<>();//new double[len][d2v.getLayerSize()];
//        INDArray oneHot = new NDArray();

        int i = 0;
        while (train.hasNext()) {
            String s = train.nextSentence();
            int ans = kv.get(train.currentLabel());
//            sentences[i]= d2v.inferVector(s).data().asDouble();
            double[] oh = new double[kv.size()];
            oh[ans] = 1.0;

            oneHot.add(oh);
            sentences.add(d2v.inferVector(s).data().asDouble());
            //model.fit(d2v.inferVector(s), new NDArray(oneHot));
            i++;
            if (i % 100 == 0) {
                System.out.println(i);
            }
        }


        model.fit(
                new NDArray(sentences.toArray(new double[][]{new double[]{0.0}})),
                new NDArray(oneHot.toArray(new double[][]{new double[]{0.0}}))
        );

        ModelSerializer.writeModel(model, "./models/NN.bin", true);
        return model;
    }

    private static void testNN(ParagraphVectors d2v, LabelAwareSentenceIterator test, HashMap<String, Integer> kv, MultiLayerNetwork model) {
        int all = 0;
        int trues = 0;
        while (test.hasNext()) {
            String sent = test.nextSentence();
            String currentLabel = test.currentLabel();
            try {
                double[] ans = model.output(d2v.inferVector(sent)).data().asDouble();
                int maxInd = 0;
                for (int j = 0; j < ans.length; j++) {
                    if (ans[j] > ans[maxInd]) {
                        maxInd = j;
                    }
                }
                if (maxInd != kv.get(currentLabel)) {
                    System.out.println(sent + " " + maxInd + " " + kv.get(currentLabel));
                } else {
//                    System.out.println(sent + " " + maxInd + " " + kv.get(currentLabel));
                    trues++;
                }
                all++;
            } catch (Exception e) {
                System.out.println("NotFound");
            }
        }
        System.out.println(all + " " + trues + " " + ((trues + 0.0) / all));
    }

    private static void testDoc2vec(ParagraphVectors build) throws IOException {
        LabelAwareSentenceIterator it = new LabelAwareListSentenceIterator(new FileInputStream("./data/test_clf.csv"), ":::::::", 1, 0);
        int all = 0;
        int trues = 0;
        HashSet<String> used = new HashSet<>();
        while (it.hasNext()) {
            String sent = it.nextSentence();
            String currentLabel = it.currentLabels().get(0);
            if (!used.contains(sent)) {
                used.add(sent);
                try {
                    String nearestLabel = build.nearestLabels(sent, 1).iterator().next();
                    if (currentLabel.equalsIgnoreCase(nearestLabel)) {
                        trues++;
                    } else {
                        System.out.println(sent + " " + currentLabel + " " + nearestLabel);
                    }
                    all++;
                } catch (Exception e) {
//                e.printStackTrace();
//                return;
                }
            }

        }
        System.out.println("all:" + all);
        System.out.println("trues:" + trues);
        System.out.println("trues/all:" + ((trues + 0.0) / all));
    }

    @NotNull
    private static ParagraphVectors trainDoc2Vec() throws IOException {
        LabelAwareSentenceIterator it = new LabelAwareListSentenceIterator(new FileInputStream("./data/train_clf.csv"), ":::::::", 1, 0);

//        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
//4190 Sequences/sec
        /*
        14:03:15.246 [VectorCalculationsThread 1] INFO  o.d.m.s.SequenceVectors - Epoch: [1]; Words vectorized so far: [954956];  Lines vectorized so far: [100000]; Seq/sec: [3550,13]; Words/sec: [33902,16]; learningRate: [0.006557565110986477]
        */
        ParagraphVectors build = new ParagraphVectors.Builder()
                .seed(1337)
                .learningRate(0.025)
                .minLearningRate(0.001)
                .useHierarchicSoftmax(true)
                .windowSize(10)
                .layerSize(100)
//                .minWordFrequency(10)
//                .useAdaGrad(true)  ///TODO
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(true)
                .batchSize(100)
                .epochs(20)
                .iterations(3)
                .iterate(it)
                .workers(2)
                .trainWordVectors(true)
                .tokenizerFactory(new DefaultTokenizerFactory())
                .build();

        build.fit();

        WordVectorSerializer.writeParagraphVectors(build, "models/Doc2vec_CUDA.bin");
        return build;
    }

    private static void preprocessFile(String input, String output, boolean printClass) throws IOException {
        ArrayList<Pipe> preprocessPipeList = new ArrayList<Pipe>();
        preprocessPipeList.add(new CharSequenceLowercase());
        preprocessPipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        preprocessPipeList.add(new TokenSequence2Stem());
        preprocessPipeList.add(new TokenSequenceLowercase());
        preprocessPipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));

        PrintWriter pw = new PrintWriter(new FileWriter(new File(output)), false);
        preprocessPipeList.add(new FunctionToPipe(instance -> {
            if (((TokenSequence) instance.getData()).size() != 0) {
                for (Token t : (TokenSequence) instance.getData()) {
                    pw.print(t.getText() + " ");
                }
                if (printClass) {
                    pw.print(":::::::" + ((String) instance.getTarget()).toLowerCase());
                }
                pw.println("");
            }
            return instance;
        }));

        InstanceList instances = new InstanceList(new SerialPipes(preprocessPipeList));


        Pattern p = Pattern.compile("(.*),(.)");
        instances.addThruPipe(
                new CsvIterator(
                        new FileReader(new File(input)),
                        p, 1, 2, -1
                )
        );
        pw.flush();
        pw.close();
    }
}
