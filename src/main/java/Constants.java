import java.util.HashSet;

/**
 * Created by azaz on 27.07.17.
 */
public class Constants {
    public static final HashSet<String> gram = new HashSet<String>() {{
        add("A	    ".trim());
        add("ADV	    ".trim());
        add("ADVPRO	".trim());
        add("ANUM	".trim());
        add("APRO	".trim());
        add("COM	    ".trim());
        add("CONJ	".trim());
        add("INTJ	".trim());
        add("NUM	        ".trim());
        add("PART	".trim());
        add("PR	    ".trim());
        add("S       	".trim());
        add("SPRO	    ".trim());
        add("V".trim());
    }};
    public static final Pool pool=Pool.getInstance(3);
}
