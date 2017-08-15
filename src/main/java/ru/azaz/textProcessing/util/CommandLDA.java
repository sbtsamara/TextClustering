package ru.azaz.textProcessing.util;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by azaz on 09.08.17.
 */
@Parameters(commandDescription = "run LDA model")
public class CommandLDA {
    @Parameter(names = {"-train"},description = "train model")
    public boolean train=false;

    @Parameter(names = {"--topic","--count"},description = "topic count for LDA model")
    public int topicCount;

    @Parameter(names = {"-it","--iterations"},description = "count of LDA passes")
    public int iterations=1000;

    @Parameter(names = {"--print"},description = "print word assignment")
    public boolean print=false;

    @Parameter(names = {"-i","--input"},description = "input file")
    public String fileToPocess;

    @Parameter(names = {"-o","--output"},description = "trained model destenition")
    public String outputModel;

    @Parameter(names = {"-m","--model"},description = "path to model")
    public String inputModel;

    @Parameter(names = "--eval", variableArity = true)
    public List<String> texts = new ArrayList<>();

    @Parameter(names = "--test")
    public String onFile = "";
}
