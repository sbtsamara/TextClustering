package ru.azaz.textProcessing.pipes;

import cc.mallet.pipe.Noop;
import cc.mallet.types.Instance;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;

import java.io.*;

/**
 * Created by azaz on 27.07.17.
 */
public class TokenSequence2File extends Noop {
    PrintWriter pw;
    public TokenSequence2File(String filename) throws IOException {
        pw = new PrintWriter(new FileWriter(new File(filename)));
    }

    public TokenSequence2File(PrintWriter pw) {
        this.pw = pw;
    }

    @Override

    public Instance pipe(Instance carrier) {
        long l = System.currentTimeMillis();
        TokenSequence ts = (TokenSequence) carrier.getData();
        for(Token t:ts){
            pw.print(t.getText()+" ");
        }
        pw.println("");
        System.out.println(carrier.getName() + ":  time: " + (System.currentTimeMillis() - l));
        return super.pipe(carrier);
    }
}
