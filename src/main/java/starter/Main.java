package starter;
/**
 * Created by azaz on 25.07.17.
 */

import models.LDA;
import models.W2v;
import com.medallia.word2vec.Word2VecModel;

import java.io.*;

public class Main {

    public static final File binFile = new File("/home/azaz/PycharmProjects/SBT/Models/ruscorpora_mean_hs.model.bin");
//    public static Word2VecModel binModel;


    public static void main(String[] args) throws Exception {

        LDA lda = new LDA();
        lda.LDAOnText();
//        preprocessFile("");

//        w2vBuildModel();

//        LDAOnText();
//        System.out.println("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*");
//        System.out.println("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*");
//        System.out.println("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*");

//        testW2VModel();
//        W2v w2v = new W2v();
//        w2v.W2VCluster();
    }

}
