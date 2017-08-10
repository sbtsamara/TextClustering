package ru.azaz.textProcessing.starter;
/**
 * Created by azaz on 25.07.17.
 */

import cc.mallet.topics.ParallelTopicModel;
import com.beust.jcommander.JCommander;
import ru.azaz.textProcessing.models.LDA;
import ru.azaz.textProcessing.util.CommandLDA;

import java.io.File;

public class Main {

    public static final File binFile = new File("/home/azaz/PycharmProjects/SBT/Models/ruscorpora_mean_hs.model.bin");
//    public static Word2VecModel binModel;


    public static void main(String[] args) throws Exception {
//        run("lda -train --count 50 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback --print -it 500");
//        run("lda -train --count 40 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback --print -it 500");
//        run("lda -train --count 30 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback --print -it 500");
//        run("lda -train --count 20 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback --print -it 500");
//        run("lda -train --count 10 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback --print -it 500");
        String[] command=args;
        if(command.length==0){
            command="lda --print --model ./models/feedback_50.bin --eval клиент добрый вечер просить разобраться какой образ база банка появиться двойник идентичный ф тот абсолютно идентичный паспортный дать который иметься задолжность ипотека следствие заблокировать зарплатный карта сотрудник банка карта разблокировать извиниться гарантия начать списывать счёт чей кредит чей это ошибка".split(" ");
        }

        run(command);
    }

    public static void run(String[] arg) throws Exception {
        JCommander jc = new JCommander();
        CommandLDA commandLDA = new CommandLDA();
        jc.addCommand("lda", commandLDA, "LDA");


//        jc.usage();

//        jc.parse("lda -train --count 50 -i ./data/cleanedFeedbacksUTF-8.csv -o ./models/feedback --print -it 500".split(" "));
//        jc.parse("lda --model ./models/feedback_35_send.bin --eval qweqweqwe qwe qweqwe".split(" "));
        try {
            jc.parse(arg);
        }catch (Exception e){
            jc.usage();
            System.exit(1);
        }
        if (jc.getParsedCommand().equalsIgnoreCase("lda")) {
            LDA lda = new LDA();
            ParallelTopicModel model = null;
            if (commandLDA.inputModel!=null) {
                model = ParallelTopicModel.read(new File(commandLDA.inputModel));
            }

            if (commandLDA.train) {
                if(model==null){
                    model = new ParallelTopicModel(commandLDA.topicCount);
                }
                model = lda.trainModel(model, commandLDA.topicCount, commandLDA.iterations, commandLDA.fileToPocess, commandLDA.outputModel);
            }

            if(model==null){
                jc.usage();
                System.exit(1);
            }

            if (commandLDA.print) {
                lda.printModel(model);
            }

            if(commandLDA.texts.size()!=0){
                lda.evaluateMode(model,commandLDA.texts.stream().reduce((s, s2) -> s+" "+s2).get());
            }
//            if(commandLDA.)
        }
    }

}
