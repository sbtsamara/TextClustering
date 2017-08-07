package starter;

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import com.medallia.word2vec.Searcher;
import com.medallia.word2vec.Word2VecModel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import pipes.FunctionToPipe;
import pipes.TokenSequence2Stem;
import util.Utils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.ListIterator;
import java.util.regex.Pattern;

/**
 * Created by azaz on 04.08.17.
 */
public class dl4jTest {
    private static final int HIDDEN_LAYER_CONT = 3;
    private static final int HIDDEN_LAYER_WIDTH = 5;
    static Word2VecModel word2VecModel ;
    static Searcher searcher ;
    public static void main(String[] args) throws IOException {

        word2VecModel = Word2VecModel.fromBinFile(new File("./models/w2v_02.bin"));
        searcher = word2VecModel.forSearch();
//        TrainModel();


        outputModel(searcher);
//        System.out.println(net.output());

//        System.out.println(input);


        }

    private static Searcher TrainModel() throws IOException {
        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.5)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(200).nOut(50)
                        .activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder().nIn(50).nOut(50)
                        .activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID)
                        .nIn(50).nOut(3).build())
                .backpropType(BackpropType.Standard)
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(20));

        ArrayList<ArrayList<INDArray>> input = new ArrayList<>();//Nd4j.emptyLike(Nd4j.zeros(200));
        ArrayList<INDArray> output = new ArrayList<>();

        ArrayList<Pipe> w2vPipeLine = new ArrayList<>();
        w2vPipeLine.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        w2vPipeLine.add(new TokenSequenceLowercase());
        w2vPipeLine.add(new TokenSequence2Stem());
        w2vPipeLine.add(new TokenSequenceLowercase());
        w2vPipeLine.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));

        INDArray zeros = Nd4j.zeros(200);

        w2vPipeLine.add(new FunctionToPipe((o) -> {
            ArrayList<INDArray> sent = new ArrayList<INDArray>();
            for (Token t : (TokenSequence) o.getData()) {
                try {
                    INDArray word = Nd4j.create(
                            Utils.getDoubles(
                                    searcher.getRawVector(t.getText())
                            )
                    );
                    sent.add(word);
                } catch (Searcher.UnknownWordException e) {
                    e.printStackTrace();
                }
                input.add(sent);
                String target = (String) o.getTarget();
                output.add(Nd4j.create(new double[]{
                        target.indexOf("В") == 0 ? 1 : 0,
                        target.indexOf("О") == 0 ? 1 : 0,
                        target.indexOf("В") < 0 && target.indexOf("О") < 0 ? 1 : 0
                }));
            }
            System.out.println(o.getData());
            System.out.println(o.getTarget());
            System.out.println("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*");
            return o;
        }));

        InstanceList pipeline = new InstanceList(new SerialPipes(w2vPipeLine));
        pipeline.addThruPipe(
                new CsvIterator(
//                        new FileReader(new File("head.txt")),
                        new FileReader(new File("./data/123.csv")),
                        "(.*),(.)", 1, 2, -1
                )
        );

        ListIterator<INDArray> outs = output.listIterator();
        for (ArrayList<INDArray> sentence : input) {
            INDArray out = outs.next();
            for (INDArray word : sentence) {
                net.fit(
                        word,
                        out
                );
            }
            net.rnnClearPreviousState();
        }
        ModelSerializer.writeModel(net, new File("./models/lstm_1.bin"), true);
        return searcher;
    }

    private static void outputModel(Searcher searcher) throws IOException {
        MultiLayerNetwork net= ModelSerializer.restoreMultiLayerNetwork(new File("./models/lstm_1.bin"));
        net.rnnClearPreviousState();


        ArrayList<INDArray> test = new ArrayList<>();
        for (String s :
//                "погашение_s задолженность_s заработный_a плата_s декабрь_s ндс_s облагаться_v фываыфва ".split(" ")
                "пользователь_s подпись_s необходимо_adv предоставлять_v заявление_s использование_s дополнительный_a механизм_s защита_s система_s рамка_s использование_s международный_a идентификатор_s мобильный_a абонент_s система_s бизнес_s онлайн_adv подразделение_s место_s ведение_s договор_s бизнес_s онлайн_adv находиться_v список_s дополнительный_a заявление_s пользователь_s подпись_s необходимо_adv лицо_s сформировывать_v заявление_s подтверждение_s факт_s смена_s карта_s предназначать_v подтверждение_s факт_s смена_s карта_s мобильный_a телефон_s формирование_s отправка_s обработка_s заявление_s подтверждение_s факт_s смена_s карта_s область_s навигация_s выбирать_v элемент_s услуга_s заявка_s заявление_s подтверждение_s факт_s смена_s карта_s область_s навигация_s открываться_v форма_s список_s заявление_s подтверждение_s факт_s смена_s карта_s сформировывать_v требоваться_v заявление_s подтверждение_s факт_s смена_s карта_s выполнять_v отправка_s заявление_s подтверждение_s факт_s смена_s карта_s банк_s последующий_a обработка_s результат_s выполнение_s указанный_a действие_s заявление_s подтверждение_s факт_s смена_s карта_s сформировывать_v отправлять_v банк_s обработка_s ".split(" ")
                ) {
            try {
                INDArray word = Nd4j.create(
                        Utils.getDoubles(
                                searcher.getRawVector(s)
                        )
                );
                test.add(word);
            } catch (Searcher.UnknownWordException e) {
                e.printStackTrace();
            }

        }

        for (INDArray word : test) {
            System.out.println(net.rnnTimeStep(word));
//            net.rnnClearPreviousState();
        }
    }
}



/*

*/