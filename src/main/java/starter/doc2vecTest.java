package starter;

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;
import pipes.FunctionToPipe;
import pipes.TokenSequence2Stem;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.regex.Pattern;

/**
 * Created by azaz on 07.08.17.
 */
public class doc2vecTest {

    public static boolean TRAIN = false;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        TRAIN=true;

//        preprocessFile("./data/typed.csv","./data/stammed_clf.csv",true);
//        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("./data/stammed_clf.csv")));
        ParagraphVectors build = null;
        if (TRAIN) {
            build = trainDoc2Vec();
        } else {
            build=WordVectorSerializer.readParagraphVectors("models/Doc2vec.bin");
            build.setTokenizerFactory(new DefaultTokenizerFactory());
        }

        testDoc2vec(build);

    }

    private static void testDoc2vec(ParagraphVectors build) throws IOException {
        LabelAwareSentenceIterator it = new LabelAwareListSentenceIterator(new FileInputStream("./data/test_clf.csv"), ":::::::", 1, 0);
        int all = 0;
        int trues = 0;
        HashSet<String> used = new HashSet<>();
        while (it.hasNext()) {
            String sent = it.nextSentence();
            String currentLabel = it.currentLabels().get(0);
            if(!used.contains(sent)) {
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
                .useAdaGrad(true)  ///TODO
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(true)
                .batchSize(100)
                .epochs(20)
                .iterations(3)
                .iterate(it)
                .workers(4)
                .trainWordVectors(true)
                .tokenizerFactory(new DefaultTokenizerFactory())
                .build();
        build.fit();

        WordVectorSerializer.writeParagraphVectors(build,"models/Doc2vec_CUDA.bin");
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
