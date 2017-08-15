package ru.azaz.textProcessing.pipes;

import cc.mallet.extract.StringTokenization;
import cc.mallet.pipe.Noop;
import cc.mallet.types.Instance;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Created by azaz on 27.07.17.
 */
public class Sentence2Collection extends Noop {
    Collection<String> arr;

    public Sentence2Collection(Collection<String> arr) {
        this.arr = arr;
    }

    @Override

    public Instance pipe(Instance carrier) {
//        long l = System.currentTimeMillis();
        if(carrier.getData() instanceof String) {
            arr.add((String) carrier.getData());
        }else if(carrier.getData() instanceof StringTokenization){
            ((StringTokenization)carrier.getData()).forEach(token -> arr.add(token.getText()));
        }

//        System.out.println(carrier.getName() + ":  time: " + (System.currentTimeMillis() - l));
        return super.pipe(carrier);
    }
}
