package ru.azaz.textProcessing.starter;
/**
 * Created by azaz on 25.07.17.
 */

import cc.mallet.topics.ParallelTopicModel;
import com.beust.jcommander.JCommander;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import ru.azaz.textProcessing.models.LDA;
import ru.azaz.textProcessing.models.StopWords;
import ru.azaz.textProcessing.models.W2v;
import ru.azaz.textProcessing.util.CommandLDA;
import ru.azaz.textProcessing.util.CommandStopWords;
import ru.azaz.textProcessing.util.CommandW2V;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {

//    public static final File binFile = new File("/home/azaz/PycharmProjects/SBT/Models/ruscorpora_mean_hs.model.bin");
//    public static Word2VecModel binModel;


    public static void main(String[] args) throws Exception {
//        run("lda -train --count 50 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback_1000it --print -it 1000".split(" "));
//        run("lda -train --count 40 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback_1000it --print -it 1000".split(" "));
//        run("lda -train --count 30 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback_1000it --print -it 1000".split(" "));
//        run("lda -train --count 20 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback_1000it --print -it 1000".split(" "));
//        run("lda -train --count 10 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback_1000it --print -it 1000".split(" "));

        String[] command = args;
        if (command.length == 0) {
            command = "lda --model ./models/feedback_50.bin --eval клиент добрый вечер просить разобраться какой образ база банка появиться двойник идентичный ф тот абсолютно идентичный паспортный дать который иметься задолжность ипотека следствие заблокировать зарплатный карта сотрудник банка карта разблокировать извиниться гарантия начать списывать счёт чей кредит чей это ошибка".split(" ");
        }


//        run("lda -train --print --count 50 -i ./data/subsample.csv -o ./models/subsampleNew -it 20000");
//        run("lda -train --print --count 30 -i ./data/subsample_BIG.csv -o ./models/tmp2_BIG_20 -it 1000");



        run("stopWords --generate --file ./data/subsample.csv --toFile ./sw.txt");

        if(1>0){
            System.exit(0);
        }

        System.setOut(new PrintStream(new FileOutputStream(new File("./Out/printed.csv"),false)));
        run("lda --model ./models/subsample/model_Topic:50_IT:20000.bin --test ./data/subsample1.csv");
//        run("lda -train --count 30 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback_5000it --print -it 5000".split(" "));

//        run(command);



    }

    private static void run(String arg) throws Exception {
        run(arg.split(" "));
    }

    public static void run(String[] arg) throws Exception {
        JCommander jc = new JCommander();
        CommandLDA commandLDA = new CommandLDA();
        CommandW2V commandW2V = new CommandW2V();
        CommandStopWords commandStopWords= new CommandStopWords();
        jc.addCommand("lda", commandLDA, "LDA");
        jc.addCommand("w2v", commandW2V, "W2V");
        jc.addCommand("stopwords", commandStopWords, "stopWords");
        try {
            jc.parse(arg);
        } catch (Exception e) {
            jc.usage();
            e.printStackTrace();
            System.exit(1);
        }

        if (jc.getParsedCommand().equalsIgnoreCase("lda")) {
            processLDAModel(jc, commandLDA);
        } else if (jc.getParsedCommand().equalsIgnoreCase("w2v")) {
            processW2VModel(jc, commandW2V);
        }else  if(jc.getParsedCommand().equalsIgnoreCase("stopwords")){
            processSW(jc,commandStopWords);
        }


    }

    private static void processSW(JCommander jc, CommandStopWords commandStopWords) throws FileNotFoundException {
        StopWords sw = new StopWords();
        if(commandStopWords.generate){
            sw.generate(commandStopWords.file);
        }
        if(commandStopWords.toFile!=null){
            sw.writeToFile(commandStopWords.toFile);
        }
    }

    private static void processW2VModel(JCommander jc, CommandW2V commandW2V) throws IOException, InterruptedException {
        W2v w2v = new W2v();
        Word2Vec model = null;
        if (commandW2V.inputModel != null) {
            model = WordVectorSerializer.readWord2VecModel(commandW2V.inputModel);
        } else if (commandW2V.fileToPocess != null) {
            model = w2v.w2vBuildModel(commandW2V.minWordFreq, commandW2V.iterations, commandW2V.layerSize, commandW2V.windowSize, commandW2V.outputModel, commandW2V.fileToPocess);
        } else {
            jc.usage();
            System.exit(1);
        }

        if (commandW2V.text.length() != 0) {
            w2v.testW2VModel(commandW2V.text, model);
        }
    }

    private static void processLDAModel(JCommander jc, CommandLDA commandLDA) throws Exception {
        LDA lda = new LDA();
        ParallelTopicModel model = null;
        if (commandLDA.inputModel != null) {
            model = ParallelTopicModel.read(new File(commandLDA.inputModel));
        }

        if (commandLDA.train) {
            if (model == null) {
                model = new ParallelTopicModel(commandLDA.topicCount);
            }
            model = lda.trainModel(model, commandLDA.topicCount, commandLDA.iterations, commandLDA.fileToPocess, commandLDA.outputModel);
        }

        if (model == null) {
            jc.usage();
            System.exit(1);
        }
        if (commandLDA.print) {
            lda.printModel(model);
        }
        if (commandLDA.texts.size() != 0) {
            lda.evaluateMode(model, commandLDA.texts.stream().reduce((s, s2) -> s + " " + s2).get());
        }
        if(commandLDA.onFile.length()!=0){
            lda.TestModel(model,commandLDA.onFile);
        }
    }

}
