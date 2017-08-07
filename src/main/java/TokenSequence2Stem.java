import cc.mallet.extract.StringSpan;
import cc.mallet.pipe.Noop;
import cc.mallet.types.Instance;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import org.json.JSONObject;
import ru.stachek66.nlp.mystem.holding.Factory;
import ru.stachek66.nlp.mystem.holding.MyStem;
import ru.stachek66.nlp.mystem.holding.MyStemApplicationException;
import ru.stachek66.nlp.mystem.holding.Request;
import ru.stachek66.nlp.mystem.model.Info;
import scala.Option;
import scala.collection.JavaConversions;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by azaz on 27.07.17.
 */
public class TokenSequence2Stem extends Noop {
    /*private final static MyStem mystemAnalyzer =
            new Factory("-igd --eng-gr --format json --weight")
                    .newMyStem("3.0", Option.<File>empty()).get();*/

    public Function<String, Stream<Token>> qwe = new Function<String, Stream<Token>>() {
        @Override
        public Stream<Token> apply(String s) {
            ArrayList<Token> ret = new ArrayList<>();
            Iterable<Info> result = null;
            String lexS = "";
            MyStem mystemAnalyzer = Constants.pool.getObject();
            try {
                result = JavaConversions.asJavaIterable(
                        mystemAnalyzer
                                .analyze(Request.apply(s))
                                .info()
                                .toIterable());
            } catch (MyStemApplicationException e) {
                e.printStackTrace();
            } finally {
                Constants.pool.returnObject(mystemAnalyzer);
            }

            for (final Info info : result) {
                if (info.lex().nonEmpty()) {
                    lexS = info.lex().get();
                    String gr = (String) ((JSONObject) new JSONObject(info.rawResponse()).getJSONArray("analysis").get(0)).get("gr");
                    String[] arrGr = gr.split("[,=\\(]");
                    lexS += "_" + Arrays.stream(arrGr).filter(s1 -> Constants.gram.contains(s1)).findFirst().get();
//                    System.out.println(lexS);
                    ret.add(new Token(lexS));
                }
            }
//            System.out.println(ret);
            return ret.stream();
        }
    };

    @Override
    public Instance pipe(Instance carrier) {
//13292
//        new JSONObject()
        long l = System.currentTimeMillis();
        TokenSequence ts = (TokenSequence) carrier.getData();
//        ts=new TokenSequence(ts.parallelStream().map(qwe).collect(Collectors.toList()));
        ArrayList<String> lst = (ArrayList<String>) ts.stream().map(Token::getText).collect(Collectors.toList());

        ArrayList<String> chunks = new ArrayList<>();
        if(lst.size()>0) {
            int len = 500;
            int start = 0, end = 0;
            for (start = 0, end = len; end < lst.size(); start += len, end += len) {
                chunks.add(lst.subList(start, Math.min(end, lst.size()) - 1).stream().reduce((s, s2) -> s + " " + s2).get());
            }
            chunks.add(lst.subList(start, lst.size()).stream().reduce((s, s2) -> s + " " + s2).get());
            carrier.setData(new TokenSequence(chunks.parallelStream().flatMap(qwe).collect(Collectors.toList())));
        }


        System.out.println(carrier.getName() + ":  time: " + (System.currentTimeMillis() - l));
        return super.pipe(carrier);
    }
}
