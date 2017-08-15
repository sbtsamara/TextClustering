package ru.azaz.textProcessing.util;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

@Parameters(commandDescription = "tools for stopwords")
public class CommandStopWords {
    @Parameter(names={"--generate"})
    public boolean generate=false;

    @Parameter(names = {"--file"},required = true)
    public String file;

    @Parameter(names={"--toFile","--tofile"})
    public String toFile;
//    public void generate(String str){
//        System.out.println("QWE");
//    }
}
