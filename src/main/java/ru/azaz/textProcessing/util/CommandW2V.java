package ru.azaz.textProcessing.util;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by azaz on 09.08.17.
 */
@Parameters(commandDescription = "run W2V model")
public class CommandW2V {
    @Parameter(names = {"-train"}, description = "train model")
    public boolean train = false;

    @Parameter(names = {"--size", "--length"}, description = "vector length for w2v model")
    public int layerSize=100;

    @Parameter(names = {"-it", "--iterations"}, description = "count of w2v passes")
    public int iterations=150;

    @Parameter(names = {"-i", "--input"}, description = "train file")
    public String fileToPocess=null;//"stammed.txt"

    @Parameter(names = {"-o", "--output"}, description = "trained model destination")
    public String outputModel=null;

    @Parameter(names = {"-m", "--model"}, description = "path to model")
    public String inputModel=null;

    @Parameter(names = "--minWordFreq",description = "minimum word frequency in dataset")
    public int minWordFreq=10;

    @Parameter(names = {"--windowSize","--window"},description = "context window size")
    public int windowSize=10;

    @Parameter(names = "--eval", variableArity = true)
    public String text ="";
}
