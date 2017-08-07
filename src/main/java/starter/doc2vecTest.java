package starter;

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.jetbrains.annotations.NotNull;
import pipes.FunctionToPipe;
import pipes.TokenSequence2Stem;

import java.io.*;
import java.util.ArrayList;
import java.util.regex.Pattern;

/**
 * Created by azaz on 07.08.17.
 */
public class doc2vecTest {

    public static final boolean TRAIN = true;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
//        preprocessFile("./data/typed.csv","./data/stammed_clf.csv",true);
//        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("./data/stammed_clf.csv")));
        ParagraphVectors build = null;
        if (TRAIN) {
            build = trainDoc2Vec();
        } else {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File("models_Doc2vec.bin")));
            build = (ParagraphVectors) ois.readObject();
            build.setTokenizerFactory(new DefaultTokenizerFactory());
        }

        LabelAwareSentenceIterator it = new LabelAwareListSentenceIterator(new FileInputStream("./data/test_clf.csv"), ":::::::", 1, 0);
        int all = 0;
        int trues = 0;
        while (it.hasNext()) {
            String sent = it.nextSentence();
            String currentLabel = it.currentLabels().get(0);
            try {
                String nearestLabel = build.nearestLabels(sent, 1).iterator().next();
                if (currentLabel.equalsIgnoreCase(nearestLabel)) {
                    trues++;
                } else {
                    System.out.println(sent + " " + currentLabel + " " + nearestLabel);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            all++;
        }
        System.out.println("all:" + all);
        System.out.println("trues:" + trues);
        System.out.println("trues/all:" + ((trues + 0.0) / all));

    }

    @NotNull
    private static ParagraphVectors trainDoc2Vec() throws IOException {
        LabelAwareSentenceIterator it = new LabelAwareListSentenceIterator(new FileInputStream("./data/train_clf.csv"), ":::::::", 1, 0);

        ParagraphVectors build = new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .useHierarchicSoftmax(true)
                .windowSize(10)
                .layerSize(100)
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(true)
                .batchSize(100)
                .epochs(75)
                .iterate(it)
                .workers(4)
                .trainWordVectors(true)
                .tokenizerFactory(new DefaultTokenizerFactory())
                .build();
        build.fit();

        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File("models_Doc2vec.bin")));
        oos.writeObject(build);
        oos.close();
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
                    pw.print(":::::::" + instance.getTarget());
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
