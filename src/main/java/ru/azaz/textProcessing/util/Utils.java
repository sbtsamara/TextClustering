package ru.azaz.textProcessing.util;

import javafx.util.Pair;

import java.util.Collection;
import java.util.Comparator;

/**
 * Created by azaz on 04.08.17.
 */
public class Utils {
    public static Comparator<Pair<Double, String>> pairComparator = Comparator.comparingDouble(Pair::getKey);
    public static double[] getDoubles(Collection<Double> rawVector) {
        double[] arr = new double[rawVector.size()];
        int i = 0;
        for (Double d : rawVector) {
            arr[i++] = d;
        }
        return arr;
    }
}
