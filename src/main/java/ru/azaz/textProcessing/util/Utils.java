package ru.azaz.textProcessing.util;

import java.util.Collection;

/**
 * Created by azaz on 04.08.17.
 */
public class Utils {
    public static double[] getDoubles(Collection<Double> rawVector) {
        double[] arr = new double[rawVector.size()];
        int i = 0;
        for (Double d : rawVector) {
            arr[i++] = d;
        }
        return arr;
    }
}
