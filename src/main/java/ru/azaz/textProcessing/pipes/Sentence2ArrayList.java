package ru.azaz.textProcessing.pipes;

import cc.mallet.pipe.Noop;
import cc.mallet.types.Instance;

import java.util.ArrayList;

/**
 * Created by azaz on 27.07.17.
 */
public class Sentence2ArrayList extends Noop {
    ArrayList<String> arr;

    public Sentence2ArrayList(ArrayList<String> arr) {
        this.arr = arr;
    }

    @Override

    public Instance pipe(Instance carrier) {
        long l = System.currentTimeMillis();
        arr.add((String) carrier.getData());

        System.out.println(carrier.getName() + ":  time: " + (System.currentTimeMillis() - l));
        return super.pipe(carrier);
    }
}
