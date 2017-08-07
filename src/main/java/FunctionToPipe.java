import cc.mallet.pipe.Noop;
import cc.mallet.types.Instance;

import java.util.ArrayList;
import java.util.function.Function;

/**
 * Created by azaz on 27.07.17.
 */
public class FunctionToPipe extends Noop {
    Function<Instance,Instance> f;
    boolean printTime;

    public FunctionToPipe(Function<Instance,Instance> f,boolean printTime) {
        this.printTime=printTime;
        this.f=f;
    }

    public FunctionToPipe(Function<Instance,Instance> f) {
        this(f,false);
    }

    @Override

    public Instance pipe(Instance carrier) {
        long l = System.currentTimeMillis();

        carrier=f.apply(carrier);

        if(printTime)
            System.out.println(carrier.getName() + ":  time: " + (System.currentTimeMillis() - l));
        return super.pipe(carrier);
    }
}
